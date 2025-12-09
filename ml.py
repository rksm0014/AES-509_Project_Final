import numpy as np
from scipy.special import gamma
from scipy.integrate import quad

def mlf(alpha, beta, c, fi=6):
    def K(r, alpha, beta, z):
        return r**((1-beta)/alpha) * np.exp(-r**(1/alpha)) * (r*np.sin(np.pi*(1-beta)) - z*np.sin(np.pi*(1-beta+alpha))) / (np.pi*alpha*(r**2 - 2*r*z*np.cos(np.pi*alpha) + z**2))
    
    def P(r, alpha, beta, z, eps):
        w = (eps**(1/alpha))*np.sin(r/alpha) + r*(1+(1-beta)/alpha)
        return ((eps**(1+(1-beta)/alpha))/(2*np.pi*alpha)) * ((np.exp((eps**(1/alpha))*np.cos(r/alpha)) * (np.cos(w) + 1j*np.sin(w)))) / (eps*np.exp(1j*r) - z)

    def rombint(fun, a, b, order=6, *args):
        rom = np.zeros((2, order))
        h = b - a
        rom[0, 0] = h * (fun(a, *args) + fun(b, *args)) / 2

        ipower = 1
        for i in range(1, order):
            sum = 0
            for j in range(ipower):
                sum += fun(a + h*(j+0.5), *args)
            rom[1, 0] = (rom[0, 0] + h*sum) / 2
            for k in range(i):
                rom[1, k+1] = ((4**(k+1))*rom[1, k] - rom[0, k]) / ((4**(k+1)) - 1)
            rom[0, :i+1] = rom[1, :i+1]
            ipower *= 2
            h /= 2
        return rom[0, order-1]
    
    c = np.asarray(c).flatten()
    e = np.zeros_like(c, dtype=np.complex_)
    
    if alpha <= 0 or fi <= 0:
        return e
    
    for i, z in enumerate(c):
        if alpha == 1 and beta == 1:
            e[i] = np.exp(z)
        else:
            if beta < 0:
                rc = (-2 * np.log(10**(-fi) * np.pi / (6 * (abs(beta) + 2) * (2 * abs(beta))**(abs(beta)))))**alpha
            else:
                rc = (-2 * np.log(10**(-fi) * np.pi / 6))**alpha
            
            r0 = max([1, 2*abs(z), rc])
            
            if (alpha == 1 and beta == 1):
                e[i] = np.exp(z)
            else:
                if (alpha < 1 and abs(z) <= 1) or ((1 <= alpha < 2) and abs(z) <= np.floor(20/(2.1-alpha)**(5.5-2*alpha))) or (alpha >= 2 and abs(z) <= 50):
                    oldsum = 0
                    k = 0
                    while (alpha * k + beta) <= 0:
                        k += 1
                    newsum = z**k / gamma(alpha*k + beta)
                    while newsum != oldsum:
                        oldsum = newsum
                        k += 1
                        term = z**k / gamma(alpha*k + beta)
                        newsum += term
                    e[i] = newsum
                else:
                    if alpha <= 1 and abs(z) <= np.fix(5*alpha + 10):
                        if (abs(np.angle(z)) > np.pi*alpha) and (abs(abs(np.angle(z)) - (np.pi*alpha)) > 10**(-fi)):
                            if beta <= 1:
                                e[i] = rombint(K, 0, r0, fi, alpha, beta, z)
                            else:
                                eps = 1
                                e[i] = rombint(K, eps, r0, fi, alpha, beta, z) + rombint(P, -np.pi*alpha, np.pi*alpha, fi, alpha, beta, z, eps)
                        elif (abs(np.angle(z)) < np.pi*alpha) and (abs(abs(np.angle(z)) - (np.pi*alpha)) > 10**(-fi)):
                            if beta <= 1:
                                e[i] = rombint(K, 0, r0, fi, alpha, beta, z) + (z**((1-beta)/alpha))*(np.exp(z**(1/alpha))/alpha)
                            else:
                                eps = abs(z) / 2
                                e[i] = rombint(K, eps, r0, fi, alpha, beta, z) + rombint(P, -np.pi*alpha, np.pi*alpha, fi, alpha, beta, z, eps) + (z**((1-beta)/alpha))*(np.exp(z**(1/alpha))/alpha)
                        else:
                            eps = abs(z) + 0.5
                            e[i] = rombint(K, eps, r0, fi, alpha, beta, z) + rombint(P, -np.pi*alpha, np.pi*alpha, fi, alpha, beta, z, eps)
                    else:
                        if alpha <= 1:
                            if abs(np.angle(z)) < (np.pi*alpha/2 + min(np.pi, np.pi*alpha))/2:
                                newsum = (z**((1-beta)/alpha)) * np.exp(z**(1/alpha)) / alpha
                                for k in range(1, int(fi / np.log10(abs(z)))):
                                    newsum -= (z**-k) / gamma(beta - alpha*k)
                                e[i] = newsum
                            else:
                                newsum = 0
                                for k in range(1, int(fi / np.log10(abs(z)))):
                                    newsum -= (z**-k) / gamma(beta - alpha*k)
                                e[i] = newsum
                        else:
                            if alpha >= 2:
                                m = int(alpha / 2)
                                sum = 0
                                for h in range(m+1):
                                    zn = (z**(1/(m+1))) * np.exp((2*np.pi*1j*h)/(m+1))
                                    sum += mlf(alpha/(m+1), beta, zn, fi)
                                e[i] = sum / (m+1)
                            else:
                                e[i] = (mlf(alpha/2, beta, z**(1/2), fi) + mlf(alpha/2, beta, -z**(1/2), fi)) / 2
    
    if np.isreal(c).all():
        e = e.real
    
    e = e.reshape(c.shape)
    return e

# Example usage:
alpha = 0.5
beta = 1.0
c = [1+1j, 2-1j]
fi = 6
result = mlf(alpha, beta, c, fi)
print(result)
