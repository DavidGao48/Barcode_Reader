from pyzbar import pyzbar 

'''
decode_image 
\:brief Applies pyzbar to decode an image 
\:param image the original image 
\:returns a list of barcode objects, which has the following fields: 
            
            data: the contents of the barcode, still encoded in utf-8. 
                  Grab readable content by: 
                        barcode.data.decode("utf-8") 
            type: the type of the barcode, ex. code128
            rect: the bbox around the barcode relative to the input image 

'''
def decode_image(image): 
    barcodes = list(pyzbar.decode(image))
    return barcodes
