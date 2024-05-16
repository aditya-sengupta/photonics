from datetime import datetime
from astropy.io import fits

# Copilot experiment to help me get over procrastinating on making a standard format for lantern outputs.
def create_fits_file(commands, science_images, exposure_time=10.0, dark_frame=None, output_path='/path/to/output.fits'):
    # Create a new FITS file
    fits_file = fits.HDUList()

    # Add a primary header
    primary_header = fits.Header()
    primary_header['DATE-OBS'] = datetime.now().isoformat()
    fits_file.append(fits.PrimaryHDU(header=primary_header))

    # Add camera exposure time
    fits_file[0].header['EXPTIME'] = exposure_time

    # Add dark frame image
    if dark_frame is not None:
        fits_file.append(fits.ImageHDU(data=dark_frame, name='DARK_FRAME'))

    # Add deformable mirror commands and science images
    for command, science_image in zip(commands, science_images):
        fits_file.append(fits.ImageHDU(data=command, name='COMMAND'))
        fits_file.append(fits.ImageHDU(data=science_image, name='SCIENCE_IMAGE'))

    # Save the FITS file
    fits_file.writeto(output_path, overwrite=True)
