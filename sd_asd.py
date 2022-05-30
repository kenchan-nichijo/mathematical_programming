import numpy as np
import matplotlib.pyplot as plt
import math


# generate Q and x_opt
def gen_data(n, c, seed):
	
	np.random.seed(seed)
	
	dVec = np.random.uniform(1, c, n)
	D = np.diag(dVec)
	
	D[0, 0] = 1
	D[1, 1] = c
	
	A = np.random.rand(n, n)
	P, R = np.linalg.qr(A)

	Q = P.T @ D @ P

	x_opt = np.random.rand(n, 1)

	return Q, x_opt


def f(x, x_opt, Q):
	return (x-x_opt).T @ Q @ (x-x_opt)


def grad(x, x_opt, Q):
 	return 2*Q@(x-x_opt)


# class for the steepest descent method
class SD:
	def __init__(self, x_opt, Q, epsilon):
		self.epsilon = epsilon
		self.x_opt = x_opt
		self.Q = Q
		self.f_val = []
		self.itr = []

	def f(self, x):
		return f(x, self.x_opt, self.Q)
	
	def grad(self, x):
		return grad(x, self.x_opt, self.Q)
	
	def alpha(self, g):
		return (g.T@g) / (g.T@self.Q@g) / 2

	def update(self, x_k):
		k = 0
		self.itr.append(k)
		self.f_val.append(self.f(x_k)[0])

		while np.linalg.norm(self.grad(x_k), 2) >= self.epsilon:
			g_k = self.grad(x_k)
			alpha_k = self.alpha(g_k)
			x_next = x_k - alpha_k*g_k

			k += 1
			self.itr.append(k)
			self.f_val.append(self.f(x_next)[0])
			x_k = x_next

			if k == 5000:
				break


# class for the accelerated steepest descent method
class ASD:
	def __init__(self, x_opt, Q, epsilon, alpha, rho, tau):
		self.epsilon = epsilon
		self.x_opt = x_opt
		self.Q = Q
		self.alpha_0 = alpha
		self.rho = rho
		self.tau = []
		self.tau.append(tau)
		self.f_val = []
		self.itr = []

	def f(self, x):
		return f(x, self.x_opt, self.Q)
	
	def grad(self, x):
		return grad(x, self.x_opt, self.Q)

	def backtracking(self, y, g):
		alpha = self.alpha_0
		
		while self.f(y - alpha*g) > (self.f(y) - 1/2*(np.linalg.norm(g, 2)**2)*alpha): 
			alpha = self.rho * alpha
		
		return alpha
	
	def update(self, x_k):
		y_k = x_k
		k = 0
		self.itr.append(k)
		self.f_val.append(self.f(x_k)[0])

		while np.linalg.norm(self.grad(x_k), 2) >= self.epsilon:
			g_k = self.grad(x_k)
			alpha_k = self.backtracking(y_k, g_k)
			x_next = y_k - alpha_k*g_k
			
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

			if k == 5000:
				break
			

def main():
	seed = 50
	n = 1000
	c = 1/1000
	epsilon = pow(10,-6)
	alpha = 1
	rho = 0.9
	tau = 1

	Q, x_opt = gen_data(n, c, seed)
	x_0 = np.random.rand(n, 1)

	sd = SD(x_opt, Q, epsilon)
	asd = ASD(x_opt, Q, epsilon, alpha, rho, tau)
	
	sd.update(x_0)
	asd.update(x_0)
	
	plt.plot(sd.itr, sd.f_val, "b", label=" SD")
	plt.plot(asd.itr, asd.f_val, "r", label="ASD with restart")
	plt.ylabel("Function value")
	plt.xlabel("Iteration")
	plt.legend()
	ax = plt.gca()
	ax.set_yscale('log')
	plt.show()


if __name__ == '__main__':
	main()