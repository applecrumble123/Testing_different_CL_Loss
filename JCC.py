import torch


def jcc(x,y):

    # ensure that the columns are the same to do a outer product
    x = x.view(-1,x.shape[-1])
    y = y.view(-1,y.shape[-1])
    #print(x)
    #print(x.transpose(dim0=1,dim1=0))

    #print(x.transpose(dim0=1,dim1=0).size())
    #print(y.size())


    mixy = x.transpose(dim0=1,dim1=0).mm(y).sum()

    mixx = x.transpose(dim0=1,dim1=0).mm(x).sum()

    miyy = y.transpose(dim0=1,dim1=0).mm(y).sum()
    sumv = mixx + miyy
    mix = mixy / sumv
    # the transpose will give a cross correlation matrix (d x d)
    # The purpose of the cross correlation matrix is to find the relation for all x and y
    # mean is not useful here because the JCC loss is per batch and this is calculated as 1 batch
    memix = torch.exp(-2 * mix).mean()
    return memix




# batch 4 with 128 features
x1 = torch.randn(5, 5, 4)
#print(x1)
#print(x1.size())
#x1 = torch.tensor([[1,2,3,4], [3,4,5,6]])
#print(x1.size())

# batch 4 with 128 features
x2 = torch.randn(5, 5, 4)
#x2 = torch.tensor([[5,6,7,8], [7,8,9,10]])

#JCC_loss(x1, x2)

l = jcc(x1,x2)
print(l)



