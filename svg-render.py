import xml.etree.ElementTree as ET
import pprint, sys
from PIL import Image, ImageDraw
from matplotlib import colors


def validate_rect():
	return

def draw_rect(draw, xcoord = 0, ycoord = 0, width = 0, height = 0, rx = 0, ry = 0, style = {"fill" : None, "stroke" : "Black", "stroke-width" : 1}):


	if "fill" not in style.keys():
		style["fill"] = None
	
	if "stroke" not in style.keys():
		style["stroke"] = "Black"

	if "stroke-width" not in style.keys():
		style["stroke-width"] = 1

	if rx == 0 and ry != 0:
		rx = ry

	if ry == 0 and rx != 0:
		ry = rx



	draw.line([(xcoord + rx,ycoord),(xcoord + rx,ycoord + height),(xcoord + width - rx,ycoord + height),(xcoord + width - rx ,ycoord),
		(xcoord + rx,ycoord)], fill = style["stroke"], width = style["stroke-width"], joint = "curve")
	
	draw.arc([xcoord,ycoord,xcoord + rx * 2 ,ycoord + ry * 2 ], 180, 270, style["stroke"] ,width = style["stroke-width"])
	draw.arc([xcoord,ycoord,xcoord + rx * 2 ,ycoord + ry * 2 ], 90, 180, style["stroke"] ,width = style["stroke-width"])
	draw.arc([xcoord + width - rx*2,ycoord + height-ry*2,xcoord + width,ycoord + height], 0, 90, style["stroke"] ,width = style["stroke-width"])
	draw.arc([xcoord + width - rx*2,ycoord + height-ry*2,xcoord + width,ycoord + height], 270, 0, style["stroke"] ,width = style["stroke-width"])

	if style["fill"] != None:
		ImageDraw.floodfill(im, ((xcoord + width)/2 , (ycoord + height)/2), style["fill"])

	return

tree = ET.parse('test.svg')
root = tree.getroot()

im = Image.new('RGB', (1280, 720),"White")
draw = ImageDraw.Draw(im)

fill_coll = "Red"
fill_tr = (int(colors.to_rgb(fill_coll)[0])*255,int(colors.to_rgb(fill_coll)[1])*255,int(colors.to_rgb(fill_coll)[2])*255)

style = {"fill":fill_tr,"stroke":"Green","stroke-width" : 4}
draw_rect(draw,xcoord= 25 , ycoord = 67, width = 400, height = 90, style = style, rx = 30, ry = 50)

im.save("test.png", "PNG")