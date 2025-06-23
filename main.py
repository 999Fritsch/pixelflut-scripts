from scripts import *

voronoi = voronoi_noise(height=384, width=512)
with PixelClient() as pc:
    print_image_by_array(pc, voronoi, scramble=True)
    worm = PixelWorm(pc)
    worm.run()