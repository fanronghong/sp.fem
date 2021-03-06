import unittest
import spfem.mesh
import spfem.mapping
import spfem.asm as fasm
import numpy as np
import spfem.element as felem

class MappingAffineBasicTest(unittest.TestCase):
    def setUp(self):
        self.mesh=spfem.mesh.MeshTri()
        self.mesh.refine()

class MappingAffineFinvF(MappingAffineBasicTest):
    """Check that F(invF(x))===x"""
    def runTest(self):
        mapping=spfem.mapping.MappingAffine(self.mesh)
        y=mapping.F(np.array([[1,2,3],[1,2,3]]))
        X=mapping.invF(y)

        self.assertAlmostEqual(X[0][:,0].all(),1.0)

class MappingAffineNormalOrientation2D(unittest.TestCase):
    """Check that the normal vectors are correctly oriented in 2D."""
    def runTest(self):
        m=spfem.mesh.MeshTri()
        m.refine(2)

        e=felem.ElementTriP1()

        a=fasm.AssemblerElement(m,e)

        N1=a.fasm(lambda v,n: n[0]*v,normals=True)
        N2=a.fasm(lambda v,n: n[1]*v,normals=True)

        vec1=np.ones(N1.shape[0])
        vec2=np.zeros(N1.shape[0])
        self.assertAlmostEqual(N1.dot(vec1)+N2.dot(vec2),0.0)
        self.assertAlmostEqual(N1.dot(vec2)+N2.dot(vec1),0.0)
        self.assertAlmostEqual(N1.dot(vec1)+N2.dot(vec1),0.0)
        self.assertAlmostEqual(N1.dot(vec2)+N2.dot(vec2),0.0)

class MappingAffineNormalOrientation3D(unittest.TestCase):
    """Check that the normal vectors are correctly oriented in 3D."""
    def runTest(self):
        m=spfem.mesh.MeshTet()
        m.refine(2)

        e=felem.ElementTetP1()

        a=fasm.AssemblerElement(m,e)

        N1=a.fasm(lambda v,n: n[0]*v,normals=True)
        N2=a.fasm(lambda v,n: n[1]*v,normals=True)
        N3=a.fasm(lambda v,n: n[2]*v,normals=True)

        vec1=np.ones(N1.shape[0])
        vec2=np.zeros(N1.shape[0])

        self.assertAlmostEqual(N1.dot(vec1)+N2.dot(vec1)+N3.dot(vec1),0.0)

        self.assertTrue((N1[m.p[0,:]==1.0]>=0).all())
        self.assertTrue((N2[m.p[1,:]==1.0]>=0).all())
        self.assertTrue((N3[m.p[2,:]==1.0]>=0).all())
        self.assertTrue((N1[m.p[0,:]==0.0]<=0).all())
