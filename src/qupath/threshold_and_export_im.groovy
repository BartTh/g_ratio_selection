///**
// * QuPath script to threshold a single channel of an image.
// * This can also be useful to convert a binary image into QuPath annotations.
// *
// * First written for https://forum.image.sc/t/rendering-wsi-as-overlay-on-top-of-another-wsi/52629/24?u=petebankhead
// *
// * @author Pete Bankhead
// */
//import qupath.lib.images.servers.LabeledImageServer
//
int channel = 0                              // 0-based index for the channel to threshold
double threshold = 150                    // Threshold value
int level = 2                             // 0-based resolution level for the image pyramid (choosing 0 may be slow)
// Define output resolution
double requestedPixelSize = 0.05
int requestedTileSize = 1000

def belowClass = getPathClass('Threshold')     // Class for pixels below the threshold
def aboveClass = getPathClass('Ignore*') // Class for pixels above the threshold

// Create a single-resolution server at the desired level, if required
def server = getCurrentServer()
if (level != 0) {
    server = qupath.lib.images.servers.ImageServers.pyramidalize(server, server.getDownsampleForResolution(level))
}

// Create a thresholded image
def thresholdServer = PixelClassifierTools.createThresholdServer(server, channel, threshold, belowClass, aboveClass)

// Create annotations and add to the current object hierarchy
def hierarchy = getCurrentHierarchy()
PixelClassifierTools.createAnnotationsFromPixelClassifier(hierarchy, thresholdServer, 10000, 4000)

def imageData = getCurrentImageData()

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
//def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'tiles', name.take(4) + '_' + requestedPixelSize + '_' + requestedTileSize)
//def pathLabelOutput = buildFilePath('../..', 'tiles_labels', name.take(4) + '_' + requestedPixelSize + '_' + requestedTileSize)
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'tiles', name + '_' + requestedPixelSize + '_' + requestedTileSize)
def pathLabelOutput = buildFilePath('../..', 'tiles_labels', name + '_' + requestedPixelSize + '_' + requestedTileSize)
mkdirs(pathOutput)
mkdirs(pathLabelOutput)

// Convert to downsample
double downsample = requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('Threshold', 1)      // Choose output labels (the order matters!)
    .multichannelOutput(false)  // If true, each label is a different channel (required for multiclass probability)
    .build()

// Create an exporter that requests corresponding tiles from the original & labeled image servers
new TileExporter(imageData)
    .downsample(downsample)     // Define export resolution
    .imageExtension('.jpg')     // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .labeledImageSubDir(pathLabelOutput)
    .tileSize(requestedTileSize)              // Define size of each tile, in pixels
    .labeledServer(labelServer) // Define the labeled image server to use (i.e. the one we just built)
    .annotatedTilesOnly(true)  // If true, only export tiles if there is a (labeled) annotation present
    .overlap(64)                // Define overlap, in pixel units at the export resolution
    .writeTiles(pathOutput)     // Write tiles to the specified directory

print 'Done!'
