import numpy as np
import matplotlib.pyplot as plt
import math


# generate Q and x_opt
def gen_data(n, c, seed):

	np.random.seed(seed)

	dVec = np.random.uniform(-c, c, n)
	dVec[0] = c
	dVec[1] = -c

	D = np.diag(dVec)

	A = np.random.rand(n, n)
	P, R = np.linalg.qr(A)

	Q = P.T @ D @ P

	x_opt = np.ones((n,1))

	return Q, x_opt


def f(x, x_opt, Q):
	return ((x-x_opt).T @ Q @ (x-x_opt))**2 / 4


def grad(x, x_opt, Q):
 	return (x-x_opt).T@Q@(x-x_opt) * Q@(x-x_opt)


# class for the non-linear conjugate gradient methods
class NonlinearCG:
	def __init__(self, x_opt, Q, epsilon, alpha, rho, omega):
		self.x_opt = x_opt
		self.Q = Q
		self.epsilon = epsilon
		self.alpha_0 = alpha
		self.rho = rho
		self.omega = omega
		self.f_val = [[] for i in range(4)]
		self.itr = [[] for i in range(4)]

	def f(self, x):
		return f(x, self.x_opt, self.Q)
	
	def grad(self, x):
		return grad(x, self.x_opt, self.Q)

	def backtracking(self, x, p, g, omega):
		alpha = self.alpha_0
		
		while self.f(x + alpha*p) > (self.f(x) + omega*p.T@g*alpha): 
			alpha = self.rho * alpha
		
		return alpha

	def fr_beta(self, g, gp):
		return np.linalg.norm(gp,2)**2 / (np.linalg.norm(g,2)**2)
	
	def pr_beta(self, g, gp, y):
		return (gp.T@y) / (np.linalg.norm(g,2)**2)
	
	def hs_beta(self, gp, y, p):
		return (gp.T@y) / (p.T@y)

	def dy_beta(self, gp, y, p):
		return np.linalg.norm(gp,2)**2 / (p.T@y)

	def update(self, x_k):
		x_0 = x_k

		for i in range(4):
			k = 0
			x_k = x_0
			self.itr[i].append(k)		
			self.f_val[i].append(self.f(x_k)[0])
			g_k = self.grad(x_k)
			p_k = - g_k

			while np.linalg.norm(self.grad(x_k), 2) >= self.epsilon:
				g_k = self.grad(x_k)
				alpha_k = self.backtracking(x_k, p_k, g_k, self.omega)
				x_next = x_k + alpha_k*p_k
				g_next = self.grad(x_next)
				y_k = g_next - g_k

				if i == 0:
					p_next = - g_next + self.fr_beta(g_k, g_next)*p_k
				elif i == 1:
					p_next = - g_next + self.pr_beta(g_k, g_next, y_k)*p_k
				elif i == 2:
					p_next = - g_next + self.hs_beta(g_next, y_k, p_k)*p_k
				else:
					p_next = - g_next + self.dy_beta(g_next, y_k, p_k)*p_k
					
				k += 1
				self.f_val[i].append(self.f(x_next)[0])
				self.itr[i].append(k)

				x_k = x_next
				p_k = p_next

				if k == 5000:
					break


def main():
	seed = 10
	n = 1000
	c = 10
	epsilon = pow(10,-4)
	alpha = 1
	rho = 0.9
	omega = pow(10,-4)

	Q, x_opt = gen_data(n, c, seed)
	x_0 = np.zeros((n,1))

	nonlinear_cg = NonlinearCG(x_opt, Q, epsilon, alpha, rho, omega)
	nonlinear_cg.update(x_0)	

	plt.plot(nonlinear_cg.itr[0], nonlinear_cg.f_val[0], "r", label="FR")
	plt.plot(nonlinear_cg.itr[1], nonlinear_cg.f_val[1], "b", label="PR")
	plt.plot(nonlinear_cg.itr[2], nonlinear_cg.f_val[2], "y", label="HS")
	plt.plot(nonlinear_cg.itr[3], nonlinear_cg.f_val[3], "g", label="DY")
	plt.ylabel("Function value")
	plt.xlabel("Iteration")
	plt.legend()
	ax = plt.gca()
	ax.set_yscale('log')
	plt.show()


if __name__ == '__main__':
	main()