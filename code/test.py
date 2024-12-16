def f(y, z, kword = True):
    print(y,kword)
def func(*args,**kwargs):
    f(*args,**kwargs)

func(3,4,kword=False)