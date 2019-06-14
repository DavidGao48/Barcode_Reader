# Barcode_Reader

Both process_image.py and process_video.py are scripts that can be run. Use the following commands: 

  python process_image.py -i [image_path]
  python process_video.py -v [video_path]
  
When process_video.py is run, the user sees six windows. The "original" window shows the current point of 
the original video that is being processed. The "detected!" window shows information about the last detected
barcode on the video. The other four windows are cropped versions of the original video that indicate the 
parts of the original video that the program thinks is likely to be a barcode. 
