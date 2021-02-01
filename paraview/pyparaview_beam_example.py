#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
# paraview.simple._DisableFirstRenderCameraReset()

###############
# set the scene
###############

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# Properties modified on animationScene1
animationScene1.AnimationTime = 1.0

# Properties modified on animationScene1
animationScene1.AnimationTime = 0.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 0.0

# Properties modified on animationScene1
animationScene1.AnimationTime = 1.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 1.0

# Properties modified on animationScene1
animationScene1.AnimationTime = 0.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 0.0

# Properties modified on animationScene1
animationScene1.AnimationTime = 1.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 1.0

# Properties modified on animationScene1
animationScene1.AnimationTime = 0.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 0.0

# Properties modified on animationScene1
animationScene1.AnimationTime = 1.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 1.0

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

#change interaction mode for render view
renderView1.InteractionMode = '3D'

########################################
# load pvd file for the beam time series
########################################

# create a new 'PVD Reader'
# TODO: here we can start building a function depending on the filename
director_beampvd = PVDReader(FileName='/home/jonas/gitprojects/cardillo2/director_beam_lagrange.pvd')
# director_beampvd = PVDReader(FileName='/home/jonas/gitprojects/cardillo2/director_beam_B-spline.pvd')

# rename source object
RenameSource('beam_data', director_beampvd)

# show data in view
director_beampvdDisplay = Show(director_beampvd, renderView1, 'UnstructuredGridRepresentation')

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

####################
# set first director
####################
directors = ['d1', 'd2', 'd3']
colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]

# TODO: why are indeted free lines not allowed?
for i, (di, color_di) in enumerate(zip(directors, colors)):
    # create a new 'Glyph'
    glyph = Glyph(Input=director_beampvd,
        GlyphType='Arrow')
    glyph.OrientationArray = ['POINTS', di]
    glyph.ScaleArray = ['POINTS', di]
    glyph.GlyphMode = 'All Points'
    glyph.ScaleFactor = 1
    glyph.GlyphTransform = 'Transform2'
    #
    # rename source object
    RenameSource(di, glyph)
    #
    # show data in view
    glyphDisplay = Show(glyph, renderView1, 'GeometryRepresentation')
    #
    # set scalar coloring
    ColorBy(glyphDisplay, ('POINTS', di, 'Magnitude'))
    #
    # show data in view
    glyphDisplay = Show(glyph, renderView1, 'GeometryRepresentation')
    #
    # rescale color and/or opacity maps used to include current data range
    glyphDisplay.RescaleTransferFunctionToDataRange(True, False)
    #
    # show color bar/color legend
    glyphDisplay.SetScalarBarVisibility(renderView1, True)
    #
    # get color transfer function/color map for 'di'
    diLUT = GetColorTransferFunction(di)
    #
    # get opacity transfer function/opacity map for 'di'
    diPWF = GetOpacityTransferFunction(di)
    #
    # # get active view
    # renderView1 = GetActiveViewOrCreate('RenderView')
    # # uncomment following to set a specific view size
    # # renderView1.ViewSize = [1314, 541]
    # #
    # # get layout
    # layout1 = GetLayout()
    #
    # # get display properties
    # d1Display = GetDisplayProperties(d1, view=renderView1)
    #
    # turn off scalar coloring
    ColorBy(glyphDisplay, None)
    #
    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(diLUT, renderView1)
    #
    # change solid color
    glyphDisplay.AmbientColor = color_di
    glyphDisplay.DiffuseColor = color_di
    #
    # set active source
    SetActiveSource(director_beampvd)
    #
    # hide color bar/color legend
    glyphDisplay.SetScalarBarVisibility(renderView1, False)

# reset view to fit data
renderView1.ResetCamera()