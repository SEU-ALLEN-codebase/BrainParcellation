##########################################################
#Author:          Yufeng Liu
#Create time:     2024-08-07
#Description:               
##########################################################
import numpy as np
import vtk
import SimpleITK as sitk
from file_io import load_image

def recon_mesh(img, fname_output, fname_tmp='./tmp.mhd', color=None, niter=200):
    mask = img.copy()
    mask[img > 0] = 255
    snew = sitk.GetImageFromArray(mask)
    sitk.WriteImage(snew, fname_tmp)

    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(fname_tmp)
    reader.Update()

    iso = vtk.vtkMarchingCubes()
    iso.SetInputConnection(reader.GetOutputPort())
    iso.SetValue(0, 1)
    iso.ComputeNormalsOff()
    iso.Update()
    mesh = iso.GetOutput()

    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputData(mesh)
    smoother.SetNumberOfIterations(niter)
    smoother.SetRelaxationFactor(0.1)
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOn()
    smoother.Update()
    mesh = smoother.GetOutput()

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(mesh)
    decimate.SetTargetReduction(0.95)
    decimate.PreserveTopologyOn()
    decimate.Update()
    mesh = decimate.GetOutput()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.SetFeatureAngle(100.0)
    normals.ComputePointNormalsOn()
    normals.SplittingOn()
    normals.Update()
    mesh = normals.GetOutput()

    if color is not None:
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        for _ in range(mesh.GetNumberOfPoints()):
            colors.InsertNextTypedTuple(color)
        mesh.GetPointData().SetScalars(colors)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileVersion(42)
    writer.SetFileName(fname_output)
    writer.SetInputData(mesh)
    writer.Update()

if __name__ == '__main__':
    rname = 'CP'
    parc_file = '../../output_full_r671/parc_region672.nrrd'
    
    img = load_image(parc_file)
    sub_ids = np.unique(img[img > 0])
    print(f'Subregions for region {rname} is {sub_ids}')
    for sub_id in sub_ids:
        print(f'===> Processing for subregion: {sub_id}')
        cur_img = (img == sub_id).astype(np.uint8)
        fname_output = f'{rname}-s{sub_id}.vtk'
        recon_mesh(cur_img, fname_output)
    print()
    

