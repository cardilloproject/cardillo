# state file generated using paraview version 5.10.1

# uncomment the following three lines to ensure this script works in future versions
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1085, 605]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-7.562342481517999, -9.4963096702443, -4.075526597728833]
renderView1.CameraViewUp = [0.2039806419964808, 0.2444034352572987, -0.9479761909062564]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 1.8708286933869707
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1085, 605)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVD Reader'
a_Box1pvd = PVDReader(registrationName='_Box1.pvd', FileName='\\\\wsl.localhost\\Ubuntu\\home\\lisa\\git-projects\\cardillo\\examples\\Dzhanibekov_rigid_body\\_Box1.pvd')
a_Box1pvd.CellArrays = ['v', 'Omega', 'ex', 'ey', 'ez']

# create a new 'Glyph'
glyph2 = Glyph(registrationName='Glyph2', Input=a_Box1pvd,
    GlyphType='Arrow')
glyph2.OrientationArray = ['CELLS', 'ey']
glyph2.ScaleArray = ['CELLS', 'ey']
glyph2.GlyphTransform = 'Transform2'

# create a new 'Glyph'
glyph1 = Glyph(registrationName='Glyph1', Input=a_Box1pvd,
    GlyphType='Arrow')
glyph1.OrientationArray = ['CELLS', 'ex']
glyph1.ScaleArray = ['CELLS', 'ex']
glyph1.GlyphTransform = 'Transform2'

# create a new 'Glyph'
glyph3 = Glyph(registrationName='Glyph3', Input=a_Box1pvd,
    GlyphType='Arrow')
glyph3.OrientationArray = ['CELLS', 'ez']
glyph3.ScaleArray = ['CELLS', 'ez']
glyph3.GlyphTransform = 'Transform2'

# create a new 'PVD Reader'
a_Box0pvd = PVDReader(registrationName='_Box0.pvd', FileName='\\\\wsl.localhost\\Ubuntu\\home\\lisa\\git-projects\\cardillo\\examples\\Dzhanibekov_rigid_body\\_Box0.pvd')
a_Box0pvd.CellArrays = ['normals']
a_Box0pvd.PointArrays = ['v']

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from a_Box0pvd
a_Box0pvdDisplay = Show(a_Box0pvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a_Box0pvdDisplay.Representation = 'Surface'
a_Box0pvdDisplay.ColorArrayName = [None, '']
a_Box0pvdDisplay.Opacity = 0.4
a_Box0pvdDisplay.SelectTCoordArray = 'None'
a_Box0pvdDisplay.SelectNormalArray = 'None'
a_Box0pvdDisplay.SelectTangentArray = 'None'
a_Box0pvdDisplay.OSPRayScaleArray = 'v'
a_Box0pvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
a_Box0pvdDisplay.SelectOrientationVectors = 'None'
a_Box0pvdDisplay.ScaleFactor = 0.30000000000000004
a_Box0pvdDisplay.SelectScaleArray = 'None'
a_Box0pvdDisplay.GlyphType = 'Arrow'
a_Box0pvdDisplay.GlyphTableIndexArray = 'None'
a_Box0pvdDisplay.GaussianRadius = 0.015
a_Box0pvdDisplay.SetScaleArray = ['POINTS', 'v']
a_Box0pvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
a_Box0pvdDisplay.OpacityArray = ['POINTS', 'v']
a_Box0pvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
a_Box0pvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
a_Box0pvdDisplay.PolarAxes = 'PolarAxesRepresentation'
a_Box0pvdDisplay.ScalarOpacityUnitDistance = 1.6343193994109926
a_Box0pvdDisplay.OpacityArrayName = ['POINTS', 'v']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a_Box0pvdDisplay.ScaleTransferFunction.Points = [-15.00001, 0.0, 0.5, 0.0, 15.00001, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a_Box0pvdDisplay.OpacityTransferFunction.Points = [-15.00001, 0.0, 0.5, 0.0, 15.00001, 1.0, 0.5, 0.0]

# show data from a_Box1pvd
a_Box1pvdDisplay = Show(a_Box1pvd, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
a_Box1pvdDisplay.Representation = 'Surface'
a_Box1pvdDisplay.ColorArrayName = [None, '']
a_Box1pvdDisplay.SelectTCoordArray = 'None'
a_Box1pvdDisplay.SelectNormalArray = 'None'
a_Box1pvdDisplay.SelectTangentArray = 'None'
a_Box1pvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
a_Box1pvdDisplay.SelectOrientationVectors = 'None'
a_Box1pvdDisplay.ScaleFactor = 0.1
a_Box1pvdDisplay.SelectScaleArray = 'None'
a_Box1pvdDisplay.GlyphType = 'Arrow'
a_Box1pvdDisplay.GlyphTableIndexArray = 'None'
a_Box1pvdDisplay.GaussianRadius = 0.005
a_Box1pvdDisplay.SetScaleArray = [None, '']
a_Box1pvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
a_Box1pvdDisplay.OpacityArray = [None, '']
a_Box1pvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
a_Box1pvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
a_Box1pvdDisplay.PolarAxes = 'PolarAxesRepresentation'
a_Box1pvdDisplay.ScalarOpacityUnitDistance = 0.0
a_Box1pvdDisplay.OpacityArrayName = ['CELLS', 'Omega']

# show data from glyph1
glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
glyph1Display.Representation = 'Surface'
glyph1Display.AmbientColor = [1.0, 0.0, 0.0]
glyph1Display.ColorArrayName = [None, '']
glyph1Display.DiffuseColor = [1.0, 0.0, 0.0]
glyph1Display.SelectTCoordArray = 'None'
glyph1Display.SelectNormalArray = 'None'
glyph1Display.SelectTangentArray = 'None'
glyph1Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph1Display.SelectOrientationVectors = 'None'
glyph1Display.ScaleFactor = 0.010000000149011612
glyph1Display.SelectScaleArray = 'None'
glyph1Display.GlyphType = 'Arrow'
glyph1Display.GlyphTableIndexArray = 'None'
glyph1Display.GaussianRadius = 0.0005000000074505806
glyph1Display.SetScaleArray = [None, '']
glyph1Display.ScaleTransferFunction = 'PiecewiseFunction'
glyph1Display.OpacityArray = [None, '']
glyph1Display.OpacityTransferFunction = 'PiecewiseFunction'
glyph1Display.DataAxesGrid = 'GridAxesRepresentation'
glyph1Display.PolarAxes = 'PolarAxesRepresentation'

# show data from glyph2
glyph2Display = Show(glyph2, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
glyph2Display.Representation = 'Surface'
glyph2Display.AmbientColor = [0.0, 0.6666666666666666, 0.0]
glyph2Display.ColorArrayName = [None, '']
glyph2Display.DiffuseColor = [0.0, 0.6666666666666666, 0.0]
glyph2Display.SelectTCoordArray = 'None'
glyph2Display.SelectNormalArray = 'None'
glyph2Display.SelectTangentArray = 'None'
glyph2Display.OSPRayScaleArray = 'Omega'
glyph2Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph2Display.SelectOrientationVectors = 'None'
glyph2Display.ScaleFactor = 0.010004303901223466
glyph2Display.SelectScaleArray = 'None'
glyph2Display.GlyphType = 'Arrow'
glyph2Display.GlyphTableIndexArray = 'None'
glyph2Display.GaussianRadius = 0.0005002151950611733
glyph2Display.SetScaleArray = ['POINTS', 'Omega']
glyph2Display.ScaleTransferFunction = 'PiecewiseFunction'
glyph2Display.OpacityArray = ['POINTS', 'Omega']
glyph2Display.OpacityTransferFunction = 'PiecewiseFunction'
glyph2Display.DataAxesGrid = 'GridAxesRepresentation'
glyph2Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
glyph2Display.ScaleTransferFunction.Points = [-0.029275518494948277, 0.0, 0.5, 0.0, -0.029271703213453293, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
glyph2Display.OpacityTransferFunction.Points = [-0.029275518494948277, 0.0, 0.5, 0.0, -0.029271703213453293, 1.0, 0.5, 0.0]

# show data from glyph3
glyph3Display = Show(glyph3, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
glyph3Display.Representation = 'Surface'
glyph3Display.AmbientColor = [0.0, 0.0, 1.0]
glyph3Display.ColorArrayName = [None, '']
glyph3Display.DiffuseColor = [0.0, 0.0, 1.0]
glyph3Display.SelectTCoordArray = 'None'
glyph3Display.SelectNormalArray = 'None'
glyph3Display.SelectTangentArray = 'None'
glyph3Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph3Display.SelectOrientationVectors = 'None'
glyph3Display.ScaleFactor = 0.010000000149011612
glyph3Display.SelectScaleArray = 'None'
glyph3Display.GlyphType = 'Arrow'
glyph3Display.GlyphTableIndexArray = 'None'
glyph3Display.GaussianRadius = 0.0005000000074505806
glyph3Display.SetScaleArray = [None, '']
glyph3Display.ScaleTransferFunction = 'PiecewiseFunction'
glyph3Display.OpacityArray = [None, '']
glyph3Display.OpacityTransferFunction = 'PiecewiseFunction'
glyph3Display.DataAxesGrid = 'GridAxesRepresentation'
glyph3Display.PolarAxes = 'PolarAxesRepresentation'

# ----------------------------------------------------------------
# restore active source
SetActiveSource(a_Box1pvd)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')