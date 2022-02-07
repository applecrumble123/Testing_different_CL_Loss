import torch


def jcc(x,y):

    # ensure that the columns are the same to do a outer product
    x = x.view(-1,x.shape[-1])
    y = y.view(-1,y.shape[-1])
    #print(x)
    #print(x.transpose(dim0=1,dim1=0))

    #print(x.transpose(dim0=1,dim1=0).size())
    #print(y.size())

    # row wise mean as it is per batch
    # -1 gives the last dim, which is the row
    mixy = x.transpose(dim0=1,dim1=0).mm(y).sum(dim=1)

    mixx = x.transpose(dim0=1,dim1=0).mm(x).sum(dim=1)

    miyy = y.transpose(dim0=1,dim1=0).mm(y).sum(dim=1)
    sumv = mixx + miyy
    mix = mixy / sumv
    # the transpose will give a cov matrix (n x n)
    # n square is the mean for the cov matrix (n x n) --> that is why it is 1/n^2, it is an element wise mean
    # therefore, it is the mean of the matrix
    memix = torch.exp(-2 * mix).mean()
    return memix





# batch 4 with 128 features
x1 = torch.randn(5, 4)
#print(x1)
#print(x1.size())
#x1 = torch.tensor([[1,2,3,4], [3,4,5,6]])
#print(x1.size())

# batch 4 with 128 features
x2 = torch.randn(5, 4)
#x2 = torch.tensor([[5,6,7,8], [7,8,9,10]])

#JCC_loss(x1, x2)

l = jcc(x1,x2)
print(l)

