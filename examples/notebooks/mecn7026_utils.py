# HELPER AND UTILITY FILES FOR THE COURSE MECN7026 : FINITE ELEMENT METHODS
# By:
# S. Kok (schalk.kok@up.ac.za),
# G.J. Jansen van Rensburg (jaco.jansenvanrensburg@wits.ac.za)




def readsectionfile(filename):
    """ Read a file containing string section names and rows of numbers

    This returns a dictionary with an entry for each section and parses
    the array entries into floats as a list of rows.
    """
    section = {}
    sectionname = None
    with open(filename) as f:
        for line in f:
            if not '**' in line:
                try:
                    row = [float(item) for item in line.split(',')]
                    section[sectionname].append(row)
                except ValueError:
                    sectionname = line.strip()
                    section[sectionname] = []
    return section


def read_input_file(filename):
    """ Read sectioned input file for a FEM problem"""

    #TODO: the proper types should be used rather than just float or int for
    #everything
    from numpy import array

    section = readsectionfile(filename)

    # initialise case dictionary
    case_dictionary = {}

    # nodal coordinates
    kw = [kw for kw in section.keys() if '*node,' in kw.lower()][0]

    ndcoor = array(section[kw])
    case_dictionary['nodes'] = ndcoor[:, 0].tolist()
    case_dictionary['coor'] = ndcoor[:, 1:]

    # elements:
    kw = [kw for kw in section.keys() if '*element,' in kw.lower()][0]

    eltype =kw.split(',')[1].split('=')[1]
    case_dictionary['eltype'] = eltype

    elnodes = array(section[kw])
    case_dictionary['elnodes'] = array(elnodes, int)

    # boundary values:
    kw = [kw for kw in section.keys() if '*boundary' in kw.lower()][0]
    case_dictionary['displ'] = sorted(section[kw])

    # check if we have a cload definition:
    kw_l = [kw for kw in section.keys() if '*cload' in kw.lower()]
    if kw_l.__len__()>0:
        case_dictionary['cload'] = sorted(section[kw_l[0]])

    # check if dload present
    kw_l = [kw for kw in section.keys() if '*dload' in kw.lower()]
    if kw_l.__len__()>0:
        # find body force direction
        kw_k = [kw for kw in section.keys() if 'grav' in kw.lower()][0]
        kw_k.split(',')
        case_dictionary['grav'] = [float(kk) for kk in kw_k.split(',') if kk.replace('.','').replace('-','').isdigit()]

    # Section details (thickness / area)
    kw_l = [kw for kw in section.keys() if '*solid section' in kw.lower()]
    if kw_l.__len__()>0:
        case_dictionary['section'] = section[kw_l[0]][0][0]

    # material information
    mat = case_dictionary['material'] = {}
    kw_l = [kw for kw in section.keys() if '*elastic' in kw.lower()]
    if kw_l.__len__()>0:
        mat['elastic'] = section[kw_l[0]][0]

    kw_l = [kw for kw in section.keys() if '*density' in kw.lower()]
    if kw_l.__len__()>0:
        mat['density'] = section[kw_l[0]][0][0]

    kw_l = [kw for kw in section.keys() if '*plastic' in kw.lower()]
    if kw_l.__len__()>0:
        mat['plastic'] = section[kw_l[0]]

    return case_dictionary# {ndcoor, nodes, coor, elnum[0], elnodes, elas, pois, t, displ, cload}
    # return (ndcoor, nodes, coor, elnum[0], elnodes, elas, pois, t, displ, cload)
    







def B_Quad8(xi, eta, X):
    import numpy as np
    D = np.matrix(np.zeros([4, 16]))  # Initialize D with zeros
    # Derivatives of shape functions wrt xi & eta
    dNdxi = 1/4.*np.matrix([[eta + 2. * xi * (1 - eta) - eta ** 2,
                             -eta + 2. * xi * (1 - eta) + eta ** 2,
                             eta + eta ** 2 + 2. * xi * (1 + eta),
                             -eta + 2. * xi * (1 + eta) - eta ** 2,
                             4*(-xi * (1 - eta)),
                             2. - 2. * eta ** 2,
                             4*(-xi * (1 + eta)),
                             -2. + 2. * eta ** 2],
                            [xi - xi ** 2 + (2. - 2. * xi) * eta,
                             -xi - xi ** 2 + (2. + 2. * xi) * eta,
                             xi + (2. + 2. * xi) * eta + xi ** 2,
                             -xi + xi ** 2 + (2. - 2. * xi) * eta,
                             -2. + 2. * xi ** 2,
                             -4 * (1 + xi) * eta,
                             2. - 2. * xi ** 2,
                             -4 * (1 - xi) * eta]])

    for i in range(2):
        for j in range(8):
            D[i, 2 * j] = dNdxi[i, j]
            D[i + 2, 2 * j + 1] = dNdxi[i, j]
              # Arrange shape function derivatives into D
    J = dNdxi * X
    detJ = np.linalg.det(J)  # Determinant of Jacobian J
    invJ = np.linalg.inv(J)  # Inverse of Jacobian
    dxidX = np.matrix([[invJ[0, 0], invJ[0, 1], 0, 0],
                       [0, 0, invJ[1, 0], invJ[1, 1]],
                       [invJ[1, 0], invJ[1, 1], invJ[0, 0], invJ[0, 1]]])

    B = dxidX * D  # Shape function derivatives wrt x and y: Eq.(2.39)
    return B, detJ


def Quad8R_Stiffness(X, Cmat, t):
    import numpy as np
    Tangent = np.matrix(np.zeros([16, 16]))  # Initialize tangent with zeros
    Gauss_pos = 1./np.sqrt(3.0)  # Gauss point location

    for jGauss in range(2):    # 2 by 2 Gauss integration loops
        eta = (-1) ** (jGauss + 1) * Gauss_pos  # Natural coordinate eta
        for iGauss in range(2):
            xi = ((-1) ** (iGauss + 1)) * Gauss_pos   # Natural coordinate xi
            [B, detJ] = B_Quad8(xi, eta, X)     # B for Eq.(4.83)
            Tangent = Tangent + B.T * Cmat * B * detJ * t
    return Tangent

def Quad8_Stiffness(X, Cmat, t):
    import numpy as np
    Tangent = np.matrix(np.zeros([16, 16]))  # Initialize tangent with zeros
    W  = [5.0/9.0, 8.0/9.0, 5.0/9.0]
    GP = [-np.sqrt(0.6), 0.0, np.sqrt(0.6)]
    for jGauss in range(3):   # 3 by 3 Gauss integration loops
        eta = GP[jGauss]      # Natural coordinate eta
        for iGauss in range(3):
            xi = GP[iGauss]   # Natural coordinate xi
            [B, detJ] = B_Quad8(xi, eta, X)     # B for Eq.(4.83)
            Tangent = Tangent + B.T * Cmat * B * detJ * t * W[jGauss]*W[iGauss]
    return Tangent

def B_Quad4(xi, eta, X):
    import numpy as np
    G = np.matrix(np.zeros([4, 8]))  # Initialize D with zeros
    # Derivatives of shape functions wrt xi & eta
    dNdxi = 1/4.*np.matrix([[(eta - 1.0),
                             (1.0 - eta),
                             (1.0+eta),
                             (-1.0-eta)],
                            [(xi - 1.0),
                             (-1.0-xi),
                             (1.0+xi),
                             (1.0-xi)]])

    for i in range(2):
        for j in range(4):
            G[i, 2 * j] = dNdxi[i, j]
            G[i + 2, 2 * j + 1] = dNdxi[i, j]
              # Arrange shape function derivatives into D
    J = dNdxi * X
    detJ = np.linalg.det(J)  # Determinant of Jacobian J
    invJ = np.linalg.inv(J)  # Inverse of Jacobian
    A = np.matrix([[invJ[0, 0], invJ[0, 1], 0, 0],
                   [0, 0, invJ[1, 0], invJ[1, 1]],
                   [invJ[1, 0], invJ[1, 1], invJ[0, 0], invJ[0, 1]]])

    B = A * G  # Shape function derivatives wrt x and y: Eq.(2.39)
    return B, detJ

def Quad4_Stiffness(X, Cmat, t):
    import numpy as np
    Tangent = np.matrix(np.zeros([8, 8]))  # Initialize tangent with zeros
    Gauss_pos = 1./np.sqrt(3.0)  # Gauss point location

    for jGauss in range(2):    # 2 by 2 Gauss integration loops
        eta = (-1) ** (jGauss + 1) * Gauss_pos  # Natural coordinate eta
        for iGauss in range(2):
            xi = ((-1) ** (iGauss + 1)) * Gauss_pos   # Natural coordinate xi
            [B, detJ] = B_Quad4(xi, eta, X)     # B for Eq.(4.83)
            Tangent = Tangent + B.T * Cmat * B * detJ * t
    return Tangent

def nodal_values(elnodes, data):
    import numpy as np

    ValuesOut = np.c_[elnodes[:, 0],
         data[:, [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]]]

    factors = np.matrix([(1 + np.sqrt(3) / 2.), -0.5, -0.5,
         (1 - np.sqrt(3) / 2.)]).T

    cols = np.matrix([[2, 5, 11, 8],
                      [5, 2, 8, 11],
                      [8, 5, 11, 2],
                      [11, 2, 8, 5]])

    ValuesNode = np.matrix(np.zeros(np.shape(ValuesOut)))
    elnodes = np.matrix(elnodes)
    ValuesNode[:, 0] = elnodes[:, 0]

    for i in range(3):
        for row in cols:
            row = (row + i - 1)
            ValuesNode[:, row[0, 0]] = (ValuesOut[:,
                     [row[0, 0], row[0, 1], row[0, 2], row[0, 3]]] * factors)

    return ValuesOut, ValuesNode


def calc_von_mises(StressNode, pois, plane):
    import numpy as np

    VonMises = np.array(np.zeros([len(StressNode), 4]))
    StressNode = np.array(StressNode)
    if (plane == 1):
        for i in range(4):
            VonMises[:, i] = (StressNode[:, 1 + i * 3] ** 2 -
                StressNode[:, 1 + i * 3] * StressNode[:, 2 + i * 3] +
                StressNode[:, 2 + i * 3] ** 2 +
                3 * StressNode[:, 3 + i * 3] ** 2)
            VonMises[:, i] = VonMises[:, i] ** 0.5

    else:
        for i in range(4):
            VonMises[:, i] = ((1 - pois + pois ** 2) *
                (StressNode[:, 1 + i * 3] ** 2 + StressNode[:, 2 + i * 3] ** 2)
                 - (1 + pois - pois ** 2) * StressNode[:, 1 + i * 3] *
                 StressNode[:, 2 + i * 3] + 3 * StressNode[:, 3 + i * 3] ** 2)
            VonMises[:, i] = VonMises[:, i] ** 0.5

    return VonMises


def calc_tresca(StressNode, pois, plane):
    import numpy as np

    nelem = len(StressNode)

    Tresca = np.matrix(np.zeros([nelem, 4]))

    if plane:
        for j in range(nelem):
            for i in range(4):
                s = [[StressNode[j, 1 + i * 3], StressNode[j, 3 + i * 3], 0],
                     [StressNode[j, 3 + i * 3], StressNode[j, 2 + i * 3], 0],
                     [0, 0, 0]]
                principal = np.linalg.eig(s)
                Tresca[j, i] = np.max(principal[0]) - np.min(principal[0])
    else:
        for j in range(nelem):
            for i in range(4):
                s = [[StressNode[j, 1 + i * 3], StressNode[j, 3 + i * 3], 0],
                     [StressNode[j, 3 + i * 3], StressNode[j, 2 + i * 3], 0],
                     [0, 0, pois * (StressNode[j, 1 + i * 3] + StressNode[j, 2 + i * 3])]]
                principal = np.linalg.eig(s)
                Tresca[j, i] = np.max(principal[0]) - np.min(principal[0])
    return Tresca

def write_output_file(file_out, U, displ, Pb, nodes, elnodes, StrainOut,
    StressOut, tic):
    import numpy as np

    Uoutput = np.c_[nodes, U[::2], U[1::2]]
    Uoutput = np.array(Uoutput)
    StressOut = np.array(StressOut)
    StrainOut = np.array(StrainOut)

    displ = np.array(displ)
    SupReac = np.c_[displ[:, [0, 1]], Pb]
    SupReac = np.array(SupReac)

    fid = open(file_out, 'w')
    fid.write('OUTPUT OF PYTHON 2D SMALL STRAIN FEM IMPLEMENTATION \n')
    fid.write('\n')
    fid.write('          DISPLACEMENTS \n')
    fid.write('********************************* \n')
    fid.write('  Node      U1           U2 \n')
    fid.write('********************************* \n')
    for i in range(len(Uoutput)):
        fid.write('%5d %13.5f %13.5f \n' % tuple(Uoutput[i]))

    fid.write('\n')
    fid.write('                ELEMENT STRESSES \n')
    fid.write('***************************************************************************************************************************************************************** \n')
    fid.write('Element   S11_G1       S22_G1       S12_G1       S11_G2       S22_G2       S12_G2       S11_G3       S22_G3         S12_G3     S11_G4       S22_G4       S12_G4 \n')
    fid.write('***************************************************************************************************************************************************************** \n')
    for i in range(len(StressOut)):
        fid.write('%5d %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e \n'% tuple(StressOut[i]))

    fid.write('\n')
    fid.write('                ELEMENT STRAINS \n')
    fid.write('***************************************************************************************************************************************************************** \n')
    fid.write('Element   E11_G1       E22_G1       E12_G1       E11_G2       E22_G2       E12_G2       E11_G3       E22_G3       E12_G3       E11_G4       E22_G4       E12_G4 \n')
    fid.write('***************************************************************************************************************************************************************** \n')
    for i in range(len(StrainOut)):
        fid.write('%5d %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e \n'% tuple(StrainOut[i]))

    fid.write('\n')
    fid.write('       SUPPORT REACTIONS \n')
    fid.write('***************************** \n')
    fid.write('  Node   Dof      Magnitude \n')
    fid.write('***************************** \n')
    for i in range(len(SupReac)):
        fid.write('%5d %5d %17.5e \n' % tuple(SupReac[i]))
    fid.close

def getisomap(Scale):
    import numpy as np
    import matplotlib.colors as colors

    C1=[0.150,0.550,0.750,0.933,0.900,0.843,0.592,0.274,0.227,0.717,0.900,0.830,0.592,0.333,0.706,0.830,0.333,0.831,0.333,0.333,0.831,0.333,0.333,0.831,0.333]
    C2=[0.150,0.550,0.750,0.969,0.700,0.568,0.408,0.467,0.847,0.960,0.700,0.460,0.408,0.859,0.933,0.460,0.859,0.463,0.859,0.859,0.463,0.859,0.859,0.463,0.859]
    C3=[0.150,0.550,0.750,0.714,0.200,0.435,0.721,0.768,0.851,0.400,0.200,0.700,0.721,0.384,0.306,0.700,0.384,0.698,0.384,0.384,0.698,0.384,0.384,0.698,0.384]
    F = [0.0,0.28,0.45,0.60,0.8,0.9,1.0,1.08,1.22,1.39,1.63,1.82,2.0,2.35,2.5,2.65,3.1,3.65,4.15,5.1,5.65,6.15,7.1,7.65,8.15];
    
    Ff  = np.linspace(0.0,Scale,255)
    C1f = np.interp(Ff,F,C1)
    C2f = np.interp(Ff,F,C2)
    C3f = np.interp(Ff,F,C3)
    
    C1f = C1f.tolist()
    C2f = C2f.tolist()
    C3f = C3f.tolist()
    
    C = [C1f,C2f,C3f]
    C = np.matrix(C)
    C = C.T
    C = C.tolist()
    
    cm = colors.ListedColormap(C)
    return cm

def PlotIso(Xvec,Yvec,Dvec,tris_new,Title,Scale):
    import matplotlib.pyplot as mpl
    import numpy as np

    Dmin = 0.0
    Dmax = max(Dvec)
    Rng = np.linspace(Dmin,Dmax,255)
    isomap = getisomap(Scale)
    mpl.figure()
    mpl.tricontourf(Xvec,Yvec,tris_new,Dvec,Rng,cmap=isomap,extend='both')
    mpl.axis('equal')
    mpl.title(Title)

#def PlotData(Data,nnodes,elnodes,nelem,coor,Title):
def PlotData(Xvec,Yvec,Dvec,tris_new,Title,Dmin,Dmax):
    import matplotlib.pyplot as mpl
    import numpy as np
    
    Rng = np.linspace(Dmin,Dmax,21)
    mpl.figure()
    mpl.tricontourf(Xvec,Yvec,tris_new,Dvec,Rng,extend='both')
    mpl.axis('equal')
    mpl.title(Title)
    mpl.colorbar(ticks=Rng)

def PlotGeom(elnodes,coor,nelem,nnodes,magfac,U):
    import matplotlib.pyplot as mpl
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import numpy as np

    dpm = np.c_[U[0:2 * nnodes:2], U[1:2 * nnodes:2]]
    dcoor = coor + magfac * dpm
    
    fig1, ax1 = mpl.subplots()
    
    patches1 = []
    for i in range(0,nelem):
        xy = coor[elnodes[i,1:5]-1,:]
        polygon = Polygon(xy, True)
        patches1.append(polygon)
    
    for i in range(0,nelem):
        xy = dcoor[elnodes[i,1:5]-1,:]
        polygon = Polygon(xy, True)
        patches1.append(polygon)
    
    p1 = PatchCollection(patches1, alpha=0.4)
    
    colr = 100*np.ones(len(patches1))
    colr[0:nelem]=colr[0:nelem]/10.0
    p1.set_array(np.array(colr))
    
    ax1.add_collection(p1)
    mpl.axis('equal')
    mpl.ion()
    mpl.show(block=False)
    mpl.title('Deformed and undeformed mesh, magnification='+str(magfac))

def PlotClearAll():
    import matplotlib.pyplot as mpl
    mpl.close('all')

def NewTri(nelem,nnodes,elnodes,coor):
    import numpy as np
    
    Data_sm = np.zeros([nnodes,1])
    for i in range(0,nelem):
        for j in range(1,5):
            Pos = elnodes[i,j]
            Data_sm[Pos-1,0] = Data_sm[Pos-1,0]+1.0
    
    NewPos = [0]*nnodes
    k = 0
    Xvec = []
    Yvec = []
    for i in range(0,nnodes):
        if Data_sm[i,0]!=0:
            Xvec.append(coor[i,0])
            Yvec.append(coor[i,1])
            NewPos[i]=k
            k = k + 1
            
    tn1 = elnodes[:,1:4]-1
    tn2 = elnodes[:,[1,3,4]]-1
    tns = tn1.tolist()+tn2.tolist()
    tris = np.asarray(tns)
    NewPos = np.array(NewPos)
    tris_new=np.zeros([2*nelem,3])
    for i in range(0,2*nelem):
        tris_new[i,:]=NewPos[tris[i,:]]
    return tris_new,Xvec,Yvec
    
def SmoothData(Data,nnodes,elnodes,nelem):
    import matplotlib.pyplot as mpl
    import numpy as np
    Data_sm = np.zeros([nnodes,2])
    for i in range(0,nelem):
        for j in range(1,5):
            Pos = elnodes[i,j]
            Data_sm[Pos-1,0] = Data_sm[Pos-1,0]+float(Data[i,j-1])
            Data_sm[Pos-1,1] = Data_sm[Pos-1,1] + 1.0
    
    Dvec = []
    NewPos = [0]*nnodes
    k = 0
    for i in range(0,nnodes):
        if Data_sm[i,1]!=0:
            Dvec.append(Data_sm[i,0]/Data_sm[i,1])
            k = k + 1
    return Dvec