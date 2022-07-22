import numpy as np
import matplotlib.pyplot as plt


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

	x_opt = np.ones((n,1))

	x_0 = np.random.rand(n,1)

	a = np.zeros((n,1))
	b = np.ones((n,1)) * 2

	return Q, x_opt, x_0, a, b


def f(x, x_opt, Q):
	return (x-x_opt).T @ Q @ (x-x_opt) / 2


def grad(x, x_opt, Q):
 	return Q@(x-x_opt)


# class for the steepest descent method
class ProjectedGD:
	def __init__(self, x_opt, Q, a, b, epsilon, alpha, rho, delta):
		self.x_opt = x_opt
		self.Q = Q
		self.a = a
		self.b = b
		self.epsilon = epsilon
		self.alpha_0 = alpha
		self.rho = rho
		self.delta = delta
		self.f_val = []
		self.itr = []

	def f(self, x):
		return f(x, self.x_opt, self.Q)
	
	def grad(self, x):
		return grad(x, self.x_opt, self.Q)

	def pi(self, v, a, b):
		omega = []
		for i in range(len(v)):
			if v[i] < a[i]:
				omega.append(a[i])
			elif a[i] <= v[i] and v[i] <= b[i]:
				omega.append(v[i])
			else:
				omega.append(b[i])

		omega = np.array(omega, dtype=float)
		return omega

	def r(self, x, g, a, b, delta):
		r = []
		for i in range(len(g)):
			if x[i] < a[i] + delta:
				r.append(min([0,g[i]]))
			elif a[i] + delta <= x[i] and x[i] <= b[i] - delta:
				r.append(g[i])
			else:
				r.append(max([0,g[i]]))

		r = np.array(r, dtype=float)
		return r
	
	def backtracking(self, x, g):
		alpha = self.alpha_0
		
		while self.f(x - alpha*g) > (self.f(x) - 0.5*(np.linalg.norm(g, 2)**2)*alpha): 
			alpha = self.rho * alpha
		
		return alpha

	def update(self, x_k):
		k = 0
		g_k = self.grad(x_k)
		d_k = - g_k
		r_k = self.r(x_k, g_k, self.a, self.b, self.delta)
		self.itr.append(k)
		self.f_val.append(self.f(x_k)[0])

		while np.linalg.norm(r_k, 2) >= self.epsilon:
			g_k = self.grad(x_k)
			d_k = - g_k
			alpha_k = self.backtracking(x_k, g_k)
			v_k = x_k + alpha_k*d_k

			x_next = self.pi(v_k, self.a, self.b)

			k += 1
			self.itr.append(k)
			self.f_val.append(self.f(x_next)[0])

			x_k = x_next
			g_k = self.grad(x_k)
			r_k = self.r(x_k, g_k, self.a, self.b, self.delta)

			if k == 5000:
				break


def main():
	seed = 10
	n = 1000
	c = 1/1000
	epsilon = pow(10,-4)
	alpha = 1
	rho = 0.9
	delta = pow(10,-8)

	Q, x_opt, x_0, a, b = gen_data(n, c, seed)

	p_gd = ProjectedGD(x_opt, Q, a, b, epsilon, alpha, rho, delta)

	p_gd.update(x_0)

	plt.plot(p_gd.itr, p_gd.f_val, "b", label="Projected gradient descent")
	
	plt.ylabel("Function value")
	plt.xlabel("Iteration")
	plt.legend()
	ax = plt.gca()
	ax.set_yscale('log')
	plt.show()


if __name__ == '__main__':
	main()