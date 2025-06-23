from scripts import *

voronoi = voronoi_noise(size=384)
with PixelClient() as pc:
    print_image_by_array(pc, voronoi)
    worm = PixelWorm(pc)
    worm.run()