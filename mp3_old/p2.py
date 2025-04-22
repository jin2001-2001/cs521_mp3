import jax
import jax.numpy as jnp
import timeit

def f(x1, x2):
    return jnp.log(x1)+x1*x2-jnp.sin(x2)

# Use jax.grad with positional argument selection
# dy_dx1 means: ∂f/∂x1
dy_dx1=jax.grad(f, argnums=0) 
dy_dx2=jax.grad(f, argnums=1)

x1_val = 2.00
x2_val = 5.00

y_val = f(x1_val, x2_val)
val_dy_dx1 = dy_dx1(x1_val, x2_val)
val_dy_dx2 = dy_dx2(x1_val, x2_val)
print("2-1 VALUE PART:\n")
print("f(x1, x2) =", y_val)
print("∂f/∂x1 =", val_dy_dx1)
print("∂f/∂x2 =", val_dy_dx2)


##2.2
jaxpr_dy_dx1 = jax.make_jaxpr(dy_dx1)(x1_val, x2_val)
jaxpr_dy_dx2 = jax.make_jaxpr(dy_dx2)(x1_val, x2_val)
print("2-2 jaxper for two gradients PART:\n")

print("jaxpr_dy_dx1\n")
print(jaxpr_dy_dx1)
print("jaxpr_dy_dx2\n")
print(jaxpr_dy_dx2)
jaxpr_ori = jax.make_jaxpr(f)(x1_val, x2_val)
print("the original imp:for original computation compare:\n")
print(jaxpr_ori)


##2.4
hlo_dy_dx1 = jax.jit(dy_dx1).lower(x1_val, x2_val).compiler_ir(dialect='hlo')
hlo_dy_dx2 = jax.jit(dy_dx2).lower(x1_val, x2_val).compiler_ir(dialect='hlo')

print("2-4 jaxper for two gradients PART:\n")
print("hlo_dy_dx1")
print(hlo_dy_dx1.as_hlo_text())
print("hlo_dy_dx2")
print(hlo_dy_dx2.as_hlo_text())


print("2-5 jaxper for two gradients PART:\n")
# Note: Use the function *handles* we defined earlier
g1 = lambda x1, x2: (jax.jit(f)(x1,x2), jax.jit(dy_dx1)(x1,x2), jax.jit(dy_dx2)(x1,x2))
g2 = jax.jit(lambda x1, x2: (f(x1,x2), dy_dx1(x1,x2), dy_dx2(x1,x2)))

print("g1_hlo:\n")
print("refer to 2.4 for hlo of dy, for f:")
print(jax.jit(f).lower(x1_val, x2_val).compiler_ir(dialect='hlo').as_hlo_text())
print("g2_hlo:\n")
print(g2.lower(x1_val, x2_val).compiler_ir(dialect='hlo').as_hlo_text())

# Warmup to trigger compilation
g1(x1_val, x2_val)
g2(x1_val, x2_val)

# Use timeit to compare runtimes
repeat = 5
number = 10000

print("Running on device:", jax.devices()[0])

time_g1 = timeit.timeit(lambda: g1(x1_val, x2_val), number=number)
time_g2 = timeit.timeit(lambda: g2(x1_val, x2_val), number=number)


##run the code before cuda-enabled lib installed, then rerun it after jax-cuda version enabled...
print(f"g1 (JIT each) avg time: {time_g1 / number * 1e6:.3f} µs")
print(f"g2 (JIT full) avg time: {time_g2 / number * 1e6:.3f} µs")



print("2-6:\n")
x1s = jnp.linspace(1.0, 10.0, 1000)
x2s = x1s + 1.0



results_a = jax.vmap(g2, in_axes=(0, 0))(x1s, x2s)
results_b = jax.vmap(g2, in_axes=(0, None))(x1s, 0.5)


jaxpr_vmap_a = jax.make_jaxpr(jax.vmap(g2, in_axes=(0, 0)))(x1s, x2s)
jaxpr_vmap_b = jax.make_jaxpr(jax.vmap(g2, in_axes=(0, None)))(x1s,0.5)

print("//--- Jaxpr for vmap (a) ---")
print(jaxpr_vmap_a)
print("//--- Jaxpr for vmap (b) ---")
print(jaxpr_vmap_b)