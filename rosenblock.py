import numpy as np
import matplotlib.pyplot as plt
import math
import time


def f(x, n):
	f = 0
	for i in range(1, n):
		f = f + 100*(x[i]-x[i-1]**2)**2 + (1-x[i])**2
	return f


def grad(x, n):
	g = []
	g.append(-400 * x[0] * (x[1]-x[0]**2))
	for i in range(1, n-1):
		g.append(200*(x[i]-x[i-1]**2) - 2*(1-x[i]) - 400 * x[i] * (x[i+1]-x[i]**2))
	g.append(200*(x[n-1]-x[n-2]**2) - 2*(1-x[n-1]))
	g = np.array(g, dtype=float)

	return g


# class for steepest descent methods
class SD:
	def __init__(self, n, epsilon, alpha, rho, omega):
		self.n = n
		self.epsilon = epsilon
		self.alpha_0 = alpha
		self.rho = rho
		self.omega = omega
		self.f_val = []
		self.itr = []

	def f(self, x):
		return f(x, self.n)
	
	def grad(self, x):
		return grad(x, self.n)
	
	def backtracking(self, x, d):
		alpha = self.alpha_0

		while self.f(x + alpha*d) > (self.f(x) - self.omega*(np.linalg.norm(d, 2)**2)*alpha): 
			alpha = self.rho * alpha
		
		return alpha

	def update(self, x_k):
		k = 0
		self.itr.append(k)
		self.f_val.append(self.f(x_k)[0])

		while np.linalg.norm(self.grad(x_k), 2) >= self.epsilon:
			g_k = self.grad(x_k)
			d_k = - g_k
			alpha_k = self.backtracking(x_k, d_k)
			x_next = x_k + alpha_k*d_k

			k += 1
			self.itr.append(k)
			self.f_val.append(self.f(x_next)[0])
			x_k = x_next

			if k == 5000:
				break


# class for accelerated steepest descent methods
class ASD:
	def __init__(self, n, epsilon, alpha, rho, tau, omega):
		self.n = n
		self.epsilon = epsilon
		self.alpha_0 = alpha
		self.rho = rho
		self.tau = []
		self.tau.append(tau)
		self.omega = omega
		self.f_val = []
		self.itr = []

	def f(self, x):
		return f(x, self.n)
	
	def grad(self, x):
		return grad(x, self.n)

	def backtracking(self, y, d):
		alpha = self.alpha_0
		
		while self.f(y + alpha*d) > (self.f(y) - self.omega*(np.linalg.norm(d, 2)**2)*alpha): 
			alpha = self.rho * alpha
		
		return alpha
	
	def update(self, x_k):
		y_k = x_k
		k = 0
		self.itr.append(k)
		self.f_val.append(self.f(x_k)[0])

		while np.linalg.norm(self.grad(x_k), 2) >= self.epsilon:
			g_k = self.grad(x_k)
			d_k = - g_k
			alpha_k = self.backtracking(y_k, -self.grad(y_k))
			x_next = y_k + alpha_k*d_k
			
			if self.f(x_next) <= self.f(x_k):
				tau_next = 1/2 * (1 + math.sqrt(1 + 4*self.tau[k]**2))
				self.tau.append(tau_next)
				y_next = x_next + ((self.tau[k]-1)/self.tau[k+1]) * (x_next-x_k)
			else:
				self.tau.append(1)
				x_next = x_k
				y_next = x_k

			k += 1
			self.itr.append(k)
			self.f_val.append(self.f(x_next)[0])
			x_k = x_next
			y_k = y_next


# class for non-linear conjugate gradient methods
class NonlinearCG:
	def __init__(self, n, epsilon, alpha, rho, omega):
		self.n = n
		self.epsilon = epsilon
		self.alpha_0 = alpha
		self.rho = rho
		self.omega = omega
		self.f_val = []
		self.itr = []

	def f(self, x):
		return f(x, self.n)
	
	def grad(self, x):
		return grad(x, self.n)

	def backtracking(self, x, p, g):
		alpha = self.alpha_0
		
		while self.f(x + alpha*p) > (self.f(x) + self.omega*p.T@g*alpha): 
			alpha = self.rho * alpha
		
		return alpha

	def dy_beta(self, gp, y, p):
		return np.linalg.norm(gp,2)**2 / (p.T@y)

	def update(self, x_k):
		x_0 = x_k
		k = 0
		x_k = x_0
		self.itr.append(k)		
		self.f_val.append(self.f(x_k)[0])
		g_k = self.grad(x_k)
		p_k = - g_k

		while np.linalg.norm(self.grad(x_k), 2) >= self.epsilon:
			g_k = self.grad(x_k)
			alpha_k = self.backtracking(x_k, p_k, g_k)
			x_next = x_k + alpha_k*p_k
			g_next = self.grad(x_next)
			y_k = g_next - g_k

			p_next = - g_next + self.dy_beta(g_next, y_k, p_k)*p_k
				
			k += 1
			self.f_val.append(self.f(x_next)[0])
			self.itr.append(k)

			x_k = x_next
			p_k = p_next


# class for quasi-Newton methods
class QuasiNewton:
	def __init__(self, n, epsilon, alpha, rho, omega):
		self.n = n
		self.epsilon = epsilon
		self.alpha_0 = alpha
		self.rho = rho
		self.omega = omega
		self.f_val = []
		self.itr = []

	def f(self, x):
		return f(x, self.n)
	
	def grad(self, x):
		return grad(x, self.n)

	def backtracking(self, x, d, g):
		alpha = self.alpha_0
		
		while self.f(x + alpha*d) > (self.f(x) + self.omega*d.T@g*alpha): 
			alpha = self.rho * alpha
		
		return alpha

	def bfgs(self, H, s, y):
		return -(s@y.T@H + H@y@s.T)/(s.T@y) + (1+(y.T@H@y)/(y.T@s))*((s@s.T)/(s.T@y))

	def update(self, x_k, H_k):
		x_0 = x_k
		H_0 = H_k

		k = 0
		x_k = x_0
		H_k = H_0
		self.itr.append(k)
		self.f_val.append(self.f(x_k)[0])
		g_k = self.grad(x_k)

		while np.linalg.norm(self.grad(x_k), 2) >= self.epsilon:
			g_k = self.grad(x_k)
			d_k = - H_k@g_k
			alpha_k = self.backtracking(x_k, d_k, g_k)
			x_next = x_k + alpha_k*d_k
			g_next = self.grad(x_next)
			s_k = x_next - x_k
			y_k = g_next - g_k

			H_next = H_k + self.bfgs(H_k, s_k, y_k)
			
			k += 1
			self.f_val.append(self.f(x_next)[0])
			self.itr.append(k)

			x_k = x_next
			H_k = H_next


def main():
	n = 10
	epsilon = pow(10,-4)
	alpha = 1
	rho = 0.9
	tau = 1
	omega = 0.25

	np.random.seed(seed=10)

	x_0 = np.random.rand(n,1) * 10
	# x_0 = np.ones((n,1)) * 2
	H_0 = np.identity(n)

	start = time.time()
	sd = SD(n, epsilon, alpha, rho, omega)
	sd.update(x_0)
	elapsed_time = time.time() - start
	print ("sd_elapsed_time:{0}".format(elapsed_time) + "[sec]")
	print("sd iteration:", sd.itr[-1])
	print("-----")

	start = time.time()
	asd = ASD(n, epsilon, alpha, rho, tau, omega)
	asd.update(x_0)
	elapsed_time = time.time() - start
	print ("asd_elapsed_time:{0}".format(elapsed_time) + "[sec]")
	print("asd iteration:", asd.itr[-1])
	print("-----")

	start = time.time()
	nonlinear_cg = NonlinearCG(n, epsilon, alpha, rho, omega)
	nonlinear_cg.update(x_0)
	elapsed_time = time.time() - start
	print ("nonlinear_elapsed_time:{0}".format(elapsed_time) + "[sec]")
	print("nonlinear_cg iteration:", nonlinear_cg.itr[-1])
	print("-----")
	
	start = time.time()
	quasi_newton = QuasiNewton(n, epsilon, alpha, rho, omega)	
	quasi_newton.update(x_0, H_0)
	elapsed_time = time.time() - start
	print ("quasi_elapsed_time:{0}".format(elapsed_time) + "[sec]")
	print("quasi_newton iteration:", quasi_newton.itr[-1])
	
	plt.plot(sd.itr, sd.f_val, "b", label=" SD")
	plt.plot(asd.itr, asd.f_val, "g", label=" ASD")
	plt.plot(nonlinear_cg.itr, nonlinear_cg.f_val, "y", label="NCG-DY")
	plt.plot(quasi_newton.itr, quasi_newton.f_val, "r", label="QNWT-BFGS")
	
	plt.ylabel("Function value")
	plt.xlabel("Iteration")
	plt.legend()
	ax = plt.gca()
	ax.set_yscale('log')
	plt.show()
	# plt.savefig('graph.svg')


if __name__ == '__main__':
	main()