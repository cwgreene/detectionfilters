import numpy

# this is setup to be one dimensional, but
# nothing stops accel, xstart, and vstart from being vectors.
def integrate(accel, xstart, vstart, dt):
    x = [xstart]
    xcur = xstart
    vcur = vstart
    for a in accel:
        xcur = .5 * dt**2 * a + vcur * dt + xcur
        vcur = dt * a + vcur
        x.append(xcur)
    return x

def add_noise(accel):
    noisy_accel = accel + numpy.random.normal(size=accel.size)
    return noisy_accel
