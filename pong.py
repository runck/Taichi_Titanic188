import taichi as ti

ti.init(arch=ti.gpu, debug=True)

n_material = 4
n_particle = ti.Vector.field(n_material, int, ())
n_particle_ = [2000, 500, 5000, 200]
n_particles = sum(n_particle_)
n_grid = 128


dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 5e-5

grid_m = ti.field(float, (n_grid, n_grid, n_material))
grid_v = ti.Vector.field(2, float, (n_grid, n_grid, n_material))
grid_doff = ti.Vector.field(2, float, (n_grid, n_grid, n_material))


m = ti.field(dtype=float, shape=n_material)  # mass 

x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
Jp = ti.field(dtype=float, shape=n_particles)  # plastic 
material = ti.field(dtype=int, shape=n_particles)  # material id


p_vol = (dx * 0.5)**2
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters

touch_freeze_1_3=ti.field(ti.i32, ())
touch_freeze_1_3[None]=False

acc = ti.Vector.field(2, dtype=float, shape=())


@ti.kernel
def substep():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0]
        grid_m[i, j, k] = 0
        
    for p in x:
        k = material[p]
        if k==2 and Jp[p]<0:
            k = material[p] = 0
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        h = max(0.1, min(5, ti.exp(10 * (1.0 - Jp[p]))))
        if k == 1:
            h = 0.3
        if k == 3:
            h = 1.0
        mu, la = mu_0 * h, lambda_0 * h
        if k == 0:
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if k == 2:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-3),
                              1 + 4.5e-4)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if k == 0:
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif k == 2:
            F[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose(
        ) + ti.Matrix.identity(float, 2) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + m[k] * C[p]
        
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset, k] += weight * (m[k] * v[p] + affine @ dpos)
            grid_m[base + offset, k] += weight * m[k]
            grid_doff[base + offset, k] += -(offset.cast(float) - fx)

    
    for i, j in ti.ndrange(n_grid, n_grid):
        if grid_m[i, j, 1]>0:
            grid_v[i, j, 1] += acc[None] * dt * 3e-3
    
        if grid_m[i, j, 1]>0 and grid_m[i, j, 0]>0:
            grid_v[i, j, 1], grid_v[i, j, 0] = \
                grid_v[i, j, 1]*0.9   + grid_v[i, j, 0]*0.1, \
                grid_v[i, j, 1]*0.09  + grid_v[i, j, 0]*0.9
            if not grid_doff[i, j, 0].y > grid_doff[i, j, 1].y:
                grid_v[i, j, 1].y += grid_m[i, j, 0] * dt *3.2e2
        
        if grid_m[i, j, 1]>0 and grid_m[i, j, 3]>0 and not touch_freeze_1_3[None]:
            di = (grid_doff[i, j, 3] - grid_doff[i, j, 1]).normalized()
            grid_v[i, j, 3] += di * dt
            touch_freeze_1_3[None]=True
        else:
            touch_freeze_1_3[None]=False

        if grid_m[i, j, 2]>0 and grid_m[i, j, 3]>0:
            grid_v[i, j, 2], grid_v[i, j, 3] = \
                grid_v[i, j, 2]*0.5 + grid_v[i, j, 3]*0.8, \
                grid_v[i, j, 2]*0.5 + grid_v[i, j, 3]*0.2

        if grid_m[i, j, 0]>0 and grid_m[i, j, 3]>0:
            grid_v[i, j, 0] = grid_v[i, j, 0] + grid_v[i, j, 3]*0.08


        
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k]  # Momentum to velocity
            
            if k!=2 and k!=3:
                grid_v[i, j, k][1] += dt * -1 * 30  # gravity

            if i < 3 and grid_v[i, j, k][0] < 0:
                grid_v[i, j, k][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j, k][0] > 0: grid_v[i, j, k][0] = 0
            if j < 3 and grid_v[i, j, k][1] < 0: grid_v[i, j, k][1] = 0
            if j > n_grid - 3 and grid_v[i, j, k][1] > 0: grid_v[i, j, k][1] = 0        
        
    for p in x:  # grid to particle (G2P)
        pad=3.0/128
        k = material[p]
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j]), k]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection
        pad=3.0/128
        if x[p].x<pad: x[p].x=pad
        if x[p].x>1-pad: x[p].x=1-pad
        if x[p].y<pad: x[p].y=pad
        if x[p].y>1-pad: x[p].y=1-pad


@ti.kernel
def init():
    n_particle[None] = n_particle_
    n_sum = 0
    
    m[0] = p_vol * 1.0
    m[1] = p_vol * 1.0
    m[2] = p_vol * 1.0
    m[3] = p_vol * 1.0
    
    for k in ti.static(range(n_material)):
        for j in range(n_particle[None][k]):
            i = n_sum+j
            
            material[i] = k
            v[i] = [0, 0]
            F[i] = ti.Matrix([[1, 0], [0, 1]])
            C[i] = ti.Matrix.zero(float, 2, 2)
            Jp[i] = 1   

            if k==0: # water
                x[i] = [0.05 + ti.random()*0.9, 0.05 + ti.random()*0.1]
            if k==1: # boat
                x[i] = [0.4 + ti.random()*0.15, 0.15 + ti.random()*0.04]
            if k==2: # snow
                x[i] = [0.15 + ti.random()*0.7, 0.75 + ti.random()*0.2]
            if k==3: # ball
                x[i] = [0.5 + ti.random()*0.05, 0.5 + ti.random()*0.05]
                v[i] = [0, -5]
            
        n_sum+=n_particle[None][k]



gui = ti.GUI("Pong!", res=512, background_color=0x112F41)
init()

tick=0
while True:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 'r': init()
        elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: break
    if gui.event is not None: acc[None] = [0, 0]
    if gui.is_pressed(ti.GUI.LEFT, 'a'): acc[None][0] = -1
    if gui.is_pressed(ti.GUI.RIGHT, 'd'): acc[None][0] = 1
    if gui.is_pressed(ti.GUI.UP, 'w'): acc[None][1] = 1e-1
    if gui.is_pressed(ti.GUI.DOWN, 's'): acc[None][1] = -1e-1
    
    for s in range(int(2e-3// dt)):
        substep()
    gui.circles(x.to_numpy(),
                radius=1.5,
                palette=[0x068587, 0xED553B, 0xEEEEF0, 0xA6B5F7, 0x3255A7, 0x6D35CB, 0xFE2E44, 0x26A5A7, 0xEDE53B],
                palette_indices=material
                )
    gui.show()
    # tick+=1
    # if tick%5 ==1:
        # gui.show(f'img/{tick:0>3d}.png')