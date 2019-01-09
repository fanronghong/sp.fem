# coding: utf-8

import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from spfem.mesh import MeshTri, MeshTet, MeshLine, MeshQuad
from spfem.assembly import AssemblerElement
from spfem.element import ElementTriP2, ElementTetP2, ElementTetP1



def TriP2Test():
    """Test triangular h-refinement.
    Also test facet assembly."""
    def U(x):
        return 1+x[0]-x[0]**2*x[1]**2

    def dUdx(x):
        return 1-2*x[0]*x[1]**2

    def dUdy(x):
        return -2*x[0]**2*x[1]

    def dudv(du,dv):
        return du[0]*dv[0]+du[1]*dv[1]

    def uv(u,v):
        return u*v

    def F(x,y):
        return 2*x**2+2*y**2

    def fv(v,x):
        return F(x[0],x[1])*v

    def G(x,y):
        return (x==1)*(3-3*y**2)+\
                (x==0)*(0)+\
                (y==1)*(1+x-3*x**2)+\
                (y==0)*(1+x)

    def gv(v,x):
        return G(x[0],x[1])*v

    dexact={}
    dexact[0]=dUdx
    dexact[1]=dUdy

    mesh=MeshTri()
    mesh.draw()
    mesh.refine(1)
    mesh.draw()
    hs=np.array([])
    H1errs=np.array([])
    L2errs=np.array([])

    for itr in range(2):
        mesh.refine()

        a=AssemblerElement(mesh,ElementTriP2())

        A=a.iasm(dudv)
        f=a.iasm(fv)

        B=a.fasm(uv)
        g=a.fasm(gv)

        u=np.zeros(a.dofnum_u.N)
        u=spsolve(A+B,f+g)

        hs=np.append(hs,mesh.param())
        L2errs=np.append(L2errs,a.L2error(u,U))
        H1errs=np.append(H1errs,a.H1error(u,dexact))

    # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.polyfit.html
    pfit=np.polyfit(np.log10(hs),np.log10(H1errs),1)

    assert (pfit[0] >= 0.95*2)


def Poisson_tetP1P2():
    """Tetrahedral refinements with P1 and P2 elements."""
    # define data
    def F(x, y, z):
        return 2 * x ** 2 + 2 * y ** 2 - 6 * x * y * z

    def G(x, y, z):
        return (x == 1) * (3 - 3 * y ** 2 + 2 * y * z ** 3) + \
               (x == 0) * (-y * z ** 3) + \
               (y == 1) * (1 + x - 3 * x ** 2 + 2 * x * z ** 3) + \
               (y == 0) * (1 + x - x * z ** 3) + \
               (z == 1) * (1 + x + 4 * x * y - x ** 2 * y ** 2) + \
               (z == 0) * (1 + x - x ** 2 * y ** 2)

    # bilinear and linear forms of the problem
    def dudv(du, dv):
        return du[0] * dv[0] + du[1] * dv[1] + du[2] * dv[2]

    def uv(u, v):
        return u * v

    def fv(v, x):
        return F(x[0], x[1], x[2]) * v

    def gv(v, x):
        return G(x[0], x[1], x[2]) * v

    # analytical solution and its derivatives
    def exact(x):
        return 1 + x[0] - x[0] ** 2 * x[1] ** 2 + x[0] * x[1] * x[2] ** 3

    dexact = {}
    dexact[0] = lambda x: 1 - 2 * x[0] * x[1] ** 2 + x[1] * x[2] ** 3
    dexact[1] = lambda x: -2 * x[0] ** 2 * x[1] + x[0] * x[2] ** 3
    dexact[2] = lambda x: 3 * x[0] * x[1] * x[2] ** 2

    # initialize arrays for saving errors
    hs1 = np.array([])
    hs2 = np.array([])

    # P1 element
    H1err1 = np.array([])
    L2err1 = np.array([])

    # P2 element
    H1err2 = np.array([])
    L2err2 = np.array([])

    # create the mesh; by default a box [0,1]^3 is meshed
    mesh = MeshTet()
    mesh.refine()

    # loop over mesh refinement levels
    for itr in range(3):
        # compute with P2 element
        b = AssemblerElement(mesh, ElementTetP2())

        # assemble the matrices and vectors related to P2
        A2 = b.iasm(dudv)
        f2 = b.iasm(fv)

        B2 = b.fasm(uv)
        g2 = b.fasm(gv)

        # initialize the solution vector and solve
        u2 = np.zeros(b.dofnum_u.N)
        u2 = spsolve(A2 + B2, f2 + g2)

        # compute error of the P2 element
        hs2 = np.append(hs2, mesh.param())
        L2err2 = np.append(L2err2, b.L2error(u2, exact))
        H1err2 = np.append(H1err2, b.H1error(u2, dexact))

        # refine mesh once
        mesh.refine()

        # create a finite element assembler = mesh + mapping + element
        a = AssemblerElement(mesh, ElementTetP1())

        # assemble the matrices and vectors related to P1
        A1 = a.iasm(dudv)
        f1 = a.iasm(fv)

        B1 = a.fasm(uv)
        g1 = a.fasm(gv)

        # initialize the solution vector and solve
        u1 = np.zeros(a.dofnum_u.N)
        u1 = spsolve(A1 + B1, f1 + g1)

        # compute errors and save them
        hs1 = np.append(hs1, mesh.param())
        L2err1 = np.append(L2err1, a.L2error(u1, exact))
        H1err1 = np.append(H1err1, a.H1error(u1, dexact))

    # create a linear fit on logarithmic scale
    pfit1 = np.polyfit(np.log10(hs1), np.log10(np.sqrt(L2err1 ** 2 + H1err1 ** 2)), 1)
    pfit2 = np.polyfit(np.log10(hs2), np.log10(np.sqrt(L2err2 ** 2 + H1err2 ** 2)), 1)

    if True:
        print("Convergence rate with P1 element: " + str(pfit1[0]))
        plt.loglog(hs1, np.sqrt(L2err1 ** 2 + H1err1 ** 2), 'bo-')
        print("Convergence rate with P2 element: " + str(pfit2[0]))
        plt.loglog(hs2, np.sqrt(L2err2 ** 2 + H1err2 ** 2), 'ro-')

    # check that convergence rates match theory
    assert (pfit1[0] >= 1)
    assert (pfit2[0] >= 2)

def example():
    mesh = MeshLine()
    mesh.refine(2)
    print(mesh.boundary_nodes())
    print(mesh.interior_nodes())
    # u = np.array([1, 5, 3, 2, 4]) # 在网格节点上的函数值，有顺序
    # mesh.plot(u)
    # plt.show()
    mapping = mesh.mapping() # 对一维没什么内容，重点看二维，三维

    tri = MeshTri(initmesh="reftri")
    tri.refine(1)
    tri.draw()
    mapping = tri.mapping() # CONTINUE

    pass

if __name__ == '__main__':
    # TriP2Test()
    # Poisson_tetP1P2()
    example()
    pass