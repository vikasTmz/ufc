import matlab.engine
import os
import cv2
import argparse
import sys



if __name__ == "__main__":

 	# Parse CLI arguments
    # When --help or no args are given, print this help
    usage_text = (
        ":"
        " " + __file__ + " "
    )

    parser = argparse.ArgumentParser(description=usage_text)

    parser.add_argument('--rgb_img', dest='rgb_img', help='RGB source image', type=str)
    parser.add_argument('--depth_img', dest='depth_img', help='Depth source image', type=str)
    parser.add_argument('--correspondence_img', dest='correspondence_img', help='Correspondence source image', type=str)
    parser.add_argument('--output_name', dest='output_name', help='output_name', type=str)

    args = parser.parse_args()

    if not args:
    	parser.print_help()
    	sys.exit()

	if (not args.rgb_img and 
		not args.depth_img and 
		not args.correspondence_img and  
		not args.output_name):
		print("Error: argument not given, aborting.")
		parser.print_help()
		sys.exit()

    for arg in vars(args):
        print('[%s] = ' % arg,  getattr(args, arg))

	print("Starting Matlab engine.....")
	eng = matlab.engine.start_matlab("-nodisplay")
	print("Matlab engine started.....")

	print("Starting geometric reconstruction.....")
	
	eng.geometric_recons(args.rgb_img, args.correspondence_img, args.depth_img, args.output_name, nargout=0)	            

	print("Stopping Matlab engine.....")
	eng.quit()
	print("Matlab engine stopped.....")
