import xml.etree.ElementTree as ET
import pprint, sys
from PIL import Image, ImageDraw
from matplotlib import colors

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return False
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

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


	if rx == ry == 0:
		draw.line([(xcoord + rx,ycoord),(xcoord + rx,ycoord + height),(xcoord + width - rx,ycoord + height),(xcoord + width - rx ,ycoord),
			(xcoord + rx,ycoord)], fill = style["stroke"], width = style["stroke-width"], joint = "curve")
	else:
		draw.line([(xcoord + rx,ycoord + height),(xcoord + width - rx,ycoord + height)], fill = style["stroke"], width = style["stroke-width"], joint = "curve")
		draw.line([(xcoord + rx,ycoord),(xcoord - rx + width,ycoord)], fill = style["stroke"], width = style["stroke-width"], joint = "curve")
		
		draw.line([(xcoord,ycoord + ry),(xcoord,ycoord + height - ry)], fill = style["stroke"], width = style["stroke-width"], joint = "curve")
		draw.line([(xcoord + width,ycoord + ry),(xcoord + width,ycoord + height - ry)], fill = style["stroke"], width = style["stroke-width"], joint = "curve")
		
		draw.arc([xcoord,ycoord,xcoord + rx * 2 ,ycoord + ry * 2 ], 180, 270, style["stroke"] ,width = style["stroke-width"])
		draw.arc([xcoord,ycoord + height - 2 * ry,xcoord + rx * 2 ,ycoord + height], 90, 180, style["stroke"] ,width = style["stroke-width"])
		draw.arc([xcoord + width - rx*2,ycoord + height-ry*2,xcoord + width,ycoord + height], 0, 90, style["stroke"] ,width = style["stroke-width"])
		draw.arc([xcoord + width - rx*2,ycoord ,xcoord + width,ycoord + ry * 2], 270, 0, style["stroke"] ,width = style["stroke-width"])

	if style["fill"] != None:
		ImageDraw.floodfill(im, ((xcoord + width)/2 , (ycoord + height)/2), style["fill"])

	return

def draw_circle(draw, cx = 0 , cy = 0, r = 0, style = {"fill" : None, "stroke" : "Black", "stroke-width" : 1}):

	if "fill" not in style.keys():
		style["fill"] = None
	
	if "stroke" not in style.keys():
		style["stroke"] = "Black"

	if "stroke-width" not in style.keys():
		style["stroke-width"] = 1

	draw.arc([cx - r,cy - r,cx + r, cy + r], 0, 360, style["stroke"] ,width = style["stroke-width"])
	
	if style["fill"] != None:
		ImageDraw.floodfill(im, (cx, cy), style["fill"])

	return 

def draw_ellipse(draw, cx = 0 , cy = 0, rx = 0, ry = 0, style = {"fill" : None, "stroke" : "Black", "stroke-width" : 1}):

	if "fill" not in style.keys():
		style["fill"] = None
	
	if "stroke" not in style.keys():
		style["stroke"] = "Black"

	if "stroke-width" not in style.keys():
		style["stroke-width"] = 1

	draw.arc([cx - rx,cy - ry,cx + rx, cy + ry], 0, 360, style["stroke"] ,width = style["stroke-width"])
	
	if style["fill"] != None:
		ImageDraw.floodfill(im, (cx, cy), style["fill"])

	return 

def draw_line(draw, x1 = 0, x2 = 0, y1 = 0, y2 = 0, style = {"fill" : None, "stroke" : "Black", "stroke-width" : 1}):

	if "fill" not in style.keys():
		style["fill"] = None
	
	if "stroke" not in style.keys():
		style["stroke"] = "Black"

	if "stroke-width" not in style.keys():
		style["stroke-width"] = 1

	draw.line([(x1,y1),(x2,y2)], fill = style["stroke"], width = style["stroke-width"], joint = "curve")

	return 

def draw_polyline(draw, points = [(0,0)], style = {"fill" : None, "stroke" : "Black", "stroke-width" : 1}):

	if "fill" not in style.keys():
		style["fill"] = None
	
	if "stroke" not in style.keys():
		style["stroke"] = "Black"

	if "stroke-width" not in style.keys():
		style["stroke-width"] = 1
	
	if style["fill"] != None:
		points.append(points[0])
		draw.line(points, fill = style["stroke"], width = style["stroke-width"], joint = "curve")

		noPint = True
		for i in range(len(points) - 2):
			for j in range(i,len(points) - 1):
				
				L1 = line(points[i],points[i+1])
				L2 = line(points[j],points[j+1])

				pint = intersection(L1,L2)
				if pint:
					x1, x2, x3 = points[i][0], points[i + 1][0], pint[0]
					y1, y2, y3 = points[i][1], points[i + 1][1], pint[1]

					onSeg = (min(x1, x2) <= x3 <= max(x1, x2)) and (min(y1, y2) <= y3 <= max(y1, y2))
					pint = (int(pint[0]),int(pint[1]))

					if onSeg and pint not in points:

						noPint = False
						ImageDraw.floodfill(im,((((((points[i+1][0] + points[j][0]) / 2 + pint[0]) /2) + pint[0])/2) + style["stroke-width"] /2 , 
							(((((points[i+1][1] + points[j][1]) / 2 + pint[1]) /2) + pint[1])/2)  + style["stroke-width"] / 2), style["fill"])
						ImageDraw.floodfill(im,((((((points[i][0] + points[j+1][0]) / 2 + pint[0]) /2) + pint[0])/2) - style["stroke-width"] / 2, 
							(((((points[i][1] + points[j+1][1]) / 2 + pint[1]) /2) + pint[1])/2) - style["stroke-width"] / 2), style["fill"])

		if noPint:

			ImageDraw.floodfill(im,(((((((points[1][0] + points[len(points) - 2][0]) / 2 + points[0][0]) /2) + points[0][0])/2) + points[0][0])/2 , 
							((((((points[1][1] + points[len(points) - 2][1]) / 2 + points[0][1]) /2) + points[0][1])/2) + points[0][1])/2), style["fill"])


	else:
		draw.line(points, fill = style["stroke"], width = style["stroke-width"], joint = "curve")




	return

tree = ET.parse('test.svg')
root = tree.getroot()

im = Image.new('RGB', (1280, 720),"White")
draw = ImageDraw.Draw(im)

fill_coll = "Red"
fill_tr = (int(colors.to_rgb(fill_coll)[0])*255,int(colors.to_rgb(fill_coll)[1])*255,int(colors.to_rgb(fill_coll)[2])*255)

style = {"fill":fill_tr,"stroke":"Green","stroke-width" : 3}

# draw_rect(draw,xcoord= 20 , ycoord = 20, width = 300, height = 90, style = style, rx = 95, ry = 15)
# draw_circle(draw, cx = 500, cy = 500, r = 100, style = style)
# draw_ellipse(draw, cx = 1000, cy = 500, rx = 100, ry = 50, style = style)
# draw_line(draw, x1 = 10, y1 = 10, x2 = 10, y2= 500, style = style)

draw_polyline(draw, [(20,120), (70,45), (70,95), (120,20), (200,50), (10,10), (100,150)], style = style)

im.save("test.png", "PNG")