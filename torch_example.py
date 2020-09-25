import torch

A = torch.tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad = True)

B = torch.tensor([1.3, -2.8], requires_grad = False) #is default anyway

C = A@B
Val = C.sum(dim = 0)
Val.backward()
print(A.grad, Val.item())


A = torch.tensor([[1, 2, 3], [2, 3, -3], [-5, 5, 6]]).float()
A[:2,:]
A[2,:]
A[:,1:]

print(A.sum(dim = 1), A.sum(dim = 0))
print(A.mean(dim = 0), A.std(dim = 1))
print(A.relu())

