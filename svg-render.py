import xml.etree.ElementTree as ET
import pprint, sys
from PIL import Image, ImageDraw, ImageOps
import PIL
from matplotlib import colors
import numpy as np
import math
import io

def getAngle(cx,cy,x,y):
    
    angle = math.atan2(cx-x, cy-y) * (180/math.pi) + 90
    if angle < 0:
        angle += 360

    return angle

def vectorAngle (ux,uy,vx,vy):
    
    sign = None 
    if ux * vy - uy * vx < 0:
        sign = -1
    else:
        sign = 1

    ua = math.sqrt(ux*ux + uy * uy)
    va = math.sqrt(vx * vx + vy * vy)
    dot = ux * vx + uy * vy

    return sign * math.acos(dot / (ua * va))

def get_center(x1,y1,x2,y2,fa,fs,rx,ry,phi):
    sinphi = math.sin(math.radians(phi))
    cosphi = math.cos(math.radians(phi))

    x = cosphi * (x1 - x2) / 2 + sinphi * (y1 - y2) / 2
    y = -sinphi * (x1 - x2) / 2 + cosphi * (y1 - y2) / 2
     
    px = x * x
    py = y * y

    prx = rx * rx 
    pry = ry * ry 

    L = px/prx + py/pry

    if L > 1:
        rx = math.sqrt(L) * abs(rx)
        ry = math.sqrt(L) * abs(ry)
    else:
        rx = abs(rx)
        ry = abs(ry)

    sign = None

    if fa == fs:
        sign = -1
    else:
        sign = 1

    M = math.sqrt(abs(((prx * pry - prx * py - pry * px) / (prx * py + pry * px)))) * sign 

    _cx = M * (rx * y) / ry
    _cy = M * -(ry * x) / rx

    cx = cosphi * _cx - sinphi * _cy + (x1 + x2) / 2
    cy = sinphi * _cx + cosphi * _cy + (y1 + y2) / 2

    theta = math.degrees(vectorAngle(1,0,(x - _cx) / rx,(y - _cy) / ry)) 

    _dTheta = math.degrees(vectorAngle((x - _cx) / rx,(y - _cy) / ry,(-x - _cx) / rx,(-y - _cy) / ry)) %360

    if fs == 0 and _dTheta > 0:
        _dTheta -= 360
    elif fs == 1 and _dTheta < 0:
        _dTheta += 360
    return cx, cy, getAngle(cx,cy,x1,y1), getAngle(cx,cy,x2,y2)


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

def getLocation(coorArr, i, j, t):
    if j == 0:
        return coorArr[i]
    return getLocation(coorArr, i, j - 1, t) * (1 - t) + getLocation(coorArr, i + 1, j - 1, t) * t

def draw_rect(im,draw, xcoord = 0, ycoord = 0, width = 0, height = 0, rx = 0, ry = 0, style = {"fill" : None, "stroke" : "Black", "stroke-width" : 1}):

    style = defaultStyle(style)

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
        ImageDraw.floodfill(im, ((xcoord + width/2) , (ycoord + height/2)), style["fill"])
    else:
        ImageDraw.floodfill(im, ((xcoord + width/2) , (ycoord + height/2)), (0,0,0))

    return im

def draw_circle(im,draw, cx = 0 , cy = 0, r = 0, style = {"fill" : None, "stroke" : "Black", "stroke-width" : 1}):

    style = defaultStyle(style)
    draw.arc([cx - r,cy - r,cx + r, cy + r], 0, 360, style["stroke"] ,width = style["stroke-width"])
    
    if style["fill"] != None:
        ImageDraw.floodfill(im, (cx, cy), style["fill"])
    else:
        ImageDraw.floodfill(im, (cx, cy), (0,0,0))

    return im

def draw_ellipse(im,draw, cx = 0 , cy = 0, rx = 0, ry = 0, style = {"fill" : None, "stroke" : "Black", "stroke-width" : 1}):

    style = defaultStyle(style)

    draw.arc([cx - rx,cy - ry,cx + rx, cy + ry], 0, 360, style["stroke"] ,width = style["stroke-width"])
    
    if style["fill"] != None:
        ImageDraw.floodfill(im, (cx, cy), style["fill"])
    else:
        ImageDraw.floodfill(im, (cx, cy), (0,0,0))
    return im

def draw_line(im,draw, x1 = 0, x2 = 0, y1 = 0, y2 = 0, style = {"fill" : None, "stroke" : "Black", "stroke-width" : 1}):
    
    style = defaultStyle(style)

    draw.line([(x1,y1),(x2,y2)], fill = style["stroke"], width = style["stroke-width"], joint = "curve")

    return im

def draw_polyline(im,draw, points = [(0,0)], style = {"fill" : None, "stroke" : "Black", "stroke-width" : 1}):

    style = defaultStyle(style)
    
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

    return im

def draw_path(im,draw, descr = None, style = {"fill" : None, "stroke" : "Black", "stroke-width" : 1}):

    if "fill" not in style.keys():
        style["fill"] = None
    
    if "stroke" not in style.keys():
        style["stroke"] = "Black"

    if "stroke-width" not in style.keys():
        style["stroke-width"] = 1

    startPoint = None
    startPointQuad = None
    lastCmd = None
    lastCmdQuad = None
    initialPoint = None

    if descr != None:
        for cmd in descr:
            if cmd[0] == "M":
                startPoint = cmd[1]
                lastCmd = None
                lastCmdQuad = None

                if initialPoint == None:
                    initialPoint = cmd[1]

                continue
            if cmd[0] == "m":
                startPoint = (startPoint[0] + cmd[1][0],startPoint[1] + cmd[1][1])
                lastCmd = None
                lastCmdQuad = None

                continue
            if cmd[0] == "L":
                draw.line([startPoint,cmd[1]], fill = style["stroke"], width = style["stroke-width"], joint = "curve")
                startPoint = cmd[1]
                lastCmd = None
                lastCmdQuad = None

                continue
            if cmd[0] == "l":
                draw.line([startPoint,(startPoint[0] + cmd[1][0],startPoint[1] + cmd[1][1])], fill = style["stroke"], width = style["stroke-width"], joint = "curve")
                startPoint = (startPoint[0] + cmd[1][0],startPoint[1] + cmd[1][1])
                lastCmdQuad = None
                lastCmd = None
                continue
            if cmd[0] == "H":
                draw.line([startPoint,(cmd[1],startPoint[1])], fill = style["stroke"], width = style["stroke-width"], joint = "curve")
                startPoint = (cmd[1],startPoint[1])
                lastCmdQuad = None
                lastCmd = None
                continue
            if cmd[0] == "h":
                draw.line([startPoint,(startPoint[0] + cmd[1],startPoint[1])], fill = style["stroke"], width = style["stroke-width"], joint = "curve")
                startPoint = (startPoint[0] + cmd[1], startPoint[1])
                lastCmdQuad = None
                lastCmd = None
                continue
            if cmd[0] == "V":
                draw.line([startPoint,(startPoint[0],cmd[1])], fill = style["stroke"], width = style["stroke-width"], joint = "curve")
                startPoint = (startPoint[0],cmd[1])
                lastCmdQuad = None
                lastCmd = None
                continue
            if cmd[0] == "v":
                draw.line([startPoint,(startPoint[0],startPoint[1] + cmd[1])], fill = style["stroke"], width = style["stroke-width"], joint = "curve")
                startPoint = (startPoint,startPoint[1] + cmd[1])
                lastCmdQuad = None
                lastCmd = None
                continue
            if cmd[0] == "C":
                numSteps = 1000
                coorArrX = [startPoint[0],cmd[1][0][0],cmd[1][1][0],cmd[1][2][0]]
                coorArrY = [startPoint[1],cmd[1][0][1],cmd[1][1][1],cmd[1][2][1]]
                
                for k in range(numSteps):
                    t = float(k) / (numSteps - 1)
                    x = int(getLocation(coorArrX, 0, 3, t))
                    y = int(getLocation(coorArrY, 0, 3, t))
                    im.putpixel((x, y), style["stroke"])
                lastCmd = (2 * cmd[1][2][0] - cmd[1][1][0],2 * cmd[1][2][1] - cmd[1][1][1])
                startPoint = (cmd[1][2][0],cmd[1][2][1])
                lastCmdQuad = None

                continue
            if cmd[0] == "c":
                numSteps = 1000
                coorArrX = [startPoint[0],startPoint[0] + cmd[1][0][0],startPoint[0] + cmd[1][1][0],startPoint[0] + cmd[1][2][0]]
                coorArrY = [startPoint[1],startPoint[1] + cmd[1][0][1],startPoint[1] + cmd[1][1][1],startPoint[1] + cmd[1][2][1]]
                
                for k in range(numSteps):
                    t = float(k) / (numSteps - 1)
                    x = int(getLocation(coorArrX, 0, 3, t))
                    y = int(getLocation(coorArrY, 0, 3, t))
                    im.putpixel((x, y), style["stroke"])
                
                lastCmd = (2 * (startPoint[0] + cmd[1][2][0]) - startPoint[0] - cmd[1][1][0] ,
                    2 * (startPoint[1] + cmd[1][2][1]) - startPoint[1] - cmd[1][1][1])
                startPoint = (startPoint[0] + cmd[1][2][0],startPoint[1] + cmd[1][2][1])
                lastCmdQuad = None

                continue
            if cmd[0] == "S":
                numSteps = 1000
                if lastCmd != None:
                    coorArrX = [startPoint[0],lastCmd[0],cmd[1][0][0],cmd[1][1][0]]
                    coorArrY = [startPoint[1],lastCmd[1],cmd[1][0][1],cmd[1][1][1]]
                else:
                    coorArrX = [startPoint[0],startPoint[0],cmd[1][0][0],cmd[1][1][0]]
                    coorArrY = [startPoint[1],startPoint[1],cmd[1][0][1],cmd[1][1][1]]

                for k in range(numSteps):
                    t = float(k) / (numSteps - 1)
                    x = int(getLocation(coorArrX, 0, 3, t))
                    y = int(getLocation(coorArrY, 0, 3, t))
                    im.putpixel((x, y), style["stroke"])

                lastCmd = (2 * cmd[1][1][0] - cmd[1][0][0],2 * cmd[1][1][1] - cmd[1][0][1])
                startPoint = (cmd[1][1][0],cmd[1][1][1])
                lastCmdQuad = None

                continue
            if cmd[0] == "s":
                numSteps = 1000
                if lastCmd != None:
                    coorArrX = [startPoint[0],lastCmd[0],startPoint[0] + cmd[1][0][0],startPoint[0] + cmd[1][1][0]]
                    coorArrY = [startPoint[1],lastCmd[1],startPoint[1] + cmd[1][0][1],startPoint[1] + cmd[1][1][1]]
                else:
                    coorArrX = [startPoint[0],startPoint[0],startPoint[0] +cmd[1][0][0],startPoint[0] +cmd[1][1][0]]
                    coorArrY = [startPoint[1],startPoint[1],startPoint[1] +cmd[1][0][1],startPoint[1] +startPoint[1] +cmd[1][1][1]]
                
                for k in range(numSteps):
                    t = float(k) / (numSteps - 1)
                    x = int(getLocation(coorArrX, 0, 3, t))
                    y = int(getLocation(coorArrY, 0, 3, t))
                    im.putpixel((x, y), style["stroke"])

                lastCmd = (2 * (startPoint[0] + cmd[1][1][0]) - startPoint[0] + cmd[1][0][0] ,
                    2 * (startPoint[1] + cmd[1][1][1]) - startPoint[1] + cmd[1][0][1])
                startPoint = (startPoint[0] + cmd[1][1][0],startPoint[1] + cmd[1][1][1])
                lastCmdQuad = None
                
                continue
            if cmd[0] == "Q":
                P = lambda t: (1 - t)**2 * np.array([startPoint[0],startPoint[1]]) + 2 * t * (1 - t) * np.array([cmd[1][0][0],
                    cmd[1][0][1]]) + t**2 * np.array([cmd[1][1][0],cmd[1][1][1]])
                points = np.array([P(t) for t in np.linspace(0, 1, 1000)])
                for point in points:
                    im.putpixel((int(point[0]),int(point[1])), style["stroke"])

                lastCmd = None
                startPoint = (cmd[1][1][0],cmd[1][1][1])
                lastCmdQuad = (2 * cmd[1][1][0] - cmd[1][0][0],2 * cmd[1][1][1] - cmd[1][0][1])

                continue

            if cmd[0] == "q":
                P = lambda t: (1 - t)**2 * np.array([startPoint[0],startPoint[1]]) + 2 * t * (1 - t) * np.array([startPoint[0] + cmd[1][0][0],
                    startPoint[1] + cmd[1][0][1]]) + t**2 * np.array([startPoint[0] + cmd[1][1][0],startPoint[1] + cmd[1][1][1]])
                points = np.array([P(t) for t in np.linspace(0, 1, 1000)])
                for point in points:
                    im.putpixel((int(point[0]),int(point[1])), style["stroke"])

                lastCmd = None
                startPoint = (startPoint[0] + cmd[1][1][0],startPoint[1] + cmd[1][1][1])
                lastCmdQuad = (2 * (startPoint[0] + cmd[1][1][0]) - cmd[1][0][0] - startPoint[0],
                    2 * (cmd[1][1][1] + startPoint[1]) - cmd[1][0][1] - startPoint[1])

                continue

            if cmd[0] == "T":
                nextPoint = None
                if lastCmdQuad != None:
                    P = lambda t: (1 - t)**2 * np.array([startPoint[0],startPoint[1]]) + 2 * t * (1 - t) * np.array([lastCmdQuad[0],
                        lastCmdQuad[1]]) + t**2 * np.array([cmd[1][0][0],cmd[1][0][1]])
                    nextPoint = [lastCmdQuad[0],lastCmdQuad[1]]
                else:
                    P = lambda t: (1 - t)**2 * np.array([startPoint[0],startPoint[1]]) + 2 * t * (1 - t) * np.array([startPoint[0],
                        startPoint[1]]) + t**2 * np.array([cmd[1][0][0],cmd[1][0][1]])
                    nextPoint = [startPoint[0],startPoint[1]]
                
                points = np.array([P(t) for t in np.linspace(0, 1, 1000)])
                for point in points:
                    im.putpixel((int(point[0]),int(point[1])), style["stroke"])

                lastCmd = None
                startPoint = (cmd[1][0][0],cmd[1][0][1])
                lastCmdQuad = (2 * cmd[1][0][0] - nextPoint[0],2 * cmd[1][0][1] - nextPoint[1])

                continue

            if cmd[0] == "t":
                nextPoint = None
                if lastCmdQuad != None:
                    P = lambda t: (1 - t)**2 * np.array([startPoint[0],startPoint[1]]) + 2 * t * (1 - t) * np.array([lastCmdQuad[0],
                        lastCmdQuad[1]]) + t**2 * np.array([startPoint[0] + cmd[1][0][0], startPoint[1] + cmd[1][0][1]])
                    nextPoint = [lastCmdQuad[0],lastCmdQuad[1]]
                else:
                    P = lambda t: (1 - t)**2 * np.array([startPoint[0],startPoint[1]]) + 2 * t * (1 - t) * np.array([startPoint[0],
                        startPoint[1]]) + t**2 * np.array([startPoint[0] + cmd[1][0][0],startPoint[1] + cmd[1][0][1]])
                    nextPoint = [startPoint[0],startPoint[1]]
                    
                points = np.array([P(t) for t in np.linspace(0, 1, 1000)])
                for point in points:
                    im.putpixel((int(point[0]),int(point[1])), style["stroke"])

                lastCmd = None
                startPoint = (startPoint[0] + cmd[1][0][0],startPoint[1] + cmd[1][0][1])
                lastCmdQuad = (2 * startPoint[0] - nextPoint[0],2 * startPoint[1]  - nextPoint[1])
                continue

            if cmd[0] == "A":
                cx,cy,p1,p2 = get_center(startPoint[0], startPoint[1],cmd[1][5][0],cmd[1][5][1],cmd[1][3],cmd[1][4],cmd[1][0],cmd[1][1],cmd[1][2])

                im2 = Image.new('RGBA', (int(cx) + 2 * cmd[1][0], int(cy) + 2 * cmd[1][1]))
                draw2 = ImageDraw.Draw(im2)
                
                if (cmd[1][3] == 1 and cmd[1][4] == 1) or (cmd[1][3] == 0 and cmd[1][4] == 1):
                    draw2.arc([cx-cmd[1][0],cy-cmd[1][1],cx+cmd[1][0],cy+cmd[1][1]],p2 + cmd[1][2],p1+ cmd[1][2] ,style["stroke"],width = style["stroke-width"])
                else:
                    draw2.arc([cx-cmd[1][0],cy-cmd[1][1],cx+cmd[1][0],cy+cmd[1][1]],p1 + cmd[1][2] ,p2 + cmd[1][2] ,style["stroke"],width = style["stroke-width"])
                im_new = im2.rotate(cmd[1][2])
                im_flip = ImageOps.flip(im_new)
                if cmd[1][3] == cmd[1][4]:
                    offset = (0, 0)
                else:
                    offset = (0, cmd[1][1])

                im.paste(im_flip,offset,mask=im_flip)

                startPoint = cmd[1][5]
                continue

            if cmd[0] == "a":
                cx,cy,p1,p2 = get_center(startPoint[0], startPoint[1],cmd[1][5][0] + startPoint[0],cmd[1][5][1] + startPoint[1],cmd[1][3],cmd[1][4],cmd[1][0],cmd[1][1],cmd[1][2])

                im2 = Image.new('RGBA', (int(cx) + 2 * cmd[1][0], int(cy) + 2 * cmd[1][1]))
                draw2 = ImageDraw.Draw(im2)
                
                if (cmd[1][3] == 1 and cmd[1][4] == 1) or (cmd[1][3] == 0 and cmd[1][4] == 1):
                    draw2.arc([cx-cmd[1][0],cy-cmd[1][1],cx+cmd[1][0],cy+cmd[1][1]],p2 + cmd[1][2],p1+ cmd[1][2] ,style["stroke"],width = style["stroke-width"])
                else:
                    draw2.arc([cx-cmd[1][0],cy-cmd[1][1],cx+cmd[1][0],cy+cmd[1][1]],p1 + cmd[1][2] ,p2 + cmd[1][2] ,style["stroke"],width = style["stroke-width"])
                im_new = im2.rotate(cmd[1][2])
                im_flip = ImageOps.flip(im_new)
                if cmd[1][3] == cmd[1][4]:
                    offset = (0, 0)
                else:
                    offset = (0, cmd[1][1])

                im.paste(im_flip,offset,mask=im_flip)

                startPoint = (cmd[1][5][0] + startPoint[0],cmd[1][5][1] + startPoint[1])
                continue 

            if cmd[0].lower() == "z":
                draw_line(draw,initialPoint[0],startPoint[0],initialPoint[1],startPoint[1],style = style)
                startPoint = None
                initialPoint = None
                continue 

def parseStyle(subItem,style):
    
    style_new = style

    if "stroke" in subItem.attrib:
        stroke_coll = subItem.attrib["stroke"].strip()
        stroke_tr = (int(colors.to_rgb(stroke_coll)[0]*255),int(colors.to_rgb(stroke_coll)[1]*255),int(colors.to_rgb(stroke_coll)[2]*255))
        style_new["stroke"] = stroke_tr

    if "fill" in subItem.attrib:
        if subItem.attrib["fill"].lower() == "none":
            style_new["fill"] = None
        else:
            fill_coll = subItem.attrib["fill"].strip()
            fill_tr = (int(colors.to_rgb(fill_coll)[0]*255),int(colors.to_rgb(fill_coll)[1]*255),int(colors.to_rgb(fill_coll)[2]*255))
            style_new["fill"] = fill_tr
    
    if "stroke-width" in subItem.attrib:
        stroke_width = int(subItem.attrib["stroke-width"].strip())
        style_new["stroke-width"] = stroke_width 

    return style_new

def defaultStyle(style):

    if "fill" not in style.keys():
        style["fill"] = None
    
    if "stroke" not in style.keys():
        style["stroke"] = "Black"

    if "stroke-width" not in style.keys():
        style["stroke-width"] = 1

    return style

def parseSVG(im,draw,root,scale,style = {"fill" : None, "stroke" : "Black", "stroke-width" : 1}):
    
    style = defaultStyle(style)
    
    for subItem in root:
        if subItem.tag.endswith("svg"):
            
            offsetX, offsetY, tranX, tranY,sizeX,sizeY = None,None,None,None,None,None

            if "viewBox" in subItem.attrib:
                sizeX = int(subItem.attrib["viewBox"].split()[2]) - int(subItem.attrib["viewBox"].split()[0])
                sizeY = int(subItem.attrib["viewBox"].split()[3]) - int(subItem.attrib["viewBox"].split()[1])
            else:
                sizeX,sizeY = im.size
            
            if "height" in subItem.attrib:
                tranY = int(subItem.attrib["height"].strip())
            else:
                tranY = im.size[1]

            if "width" in subItem.attrib:
                tranX = int(subItem.attrib["width"].strip())
            else:
                tranX = im.size[0]

            if "x" in subItem.attrib:
                offsetX = int(subItem.attrib["x"].strip())
            else:
                offsetX = 0
            
            if "y" in subItem.attrib:
                offsetY = int(subItem.attrib["y"].strip())
            else:
                offsetY = 0

            style_new = parseStyle(subItem,style)
            scale_new = (int(tranX/sizeX),int(tranY/sizeY))
            im_new = Image.new('RGB', (sizeX * scale_new[0],sizeY * scale_new[1]),"White")
            draw_new = ImageDraw.Draw(im_new)
            style_new["stroke-width"] = style_new["stroke-width"] * scale_new[0]
            im_new, draw_new = parseSVG(im_new,draw_new,subItem,scale_new,style_new)
            # im_res = im_new.resize((tranX,tranY),PIL.Image.ANTIALIAS)
            im.paste(im_new,(offsetX,offsetY))#,mask=im_new)

        if "circle" in subItem.tag:
            cx,cy,r = 0,0,0

            if "cx" in subItem.attrib:
                cx = int(subItem.attrib["cx"].strip())
            if "cy" in subItem.attrib:
                cy = int(subItem.attrib["cy"].strip())
            if "r" in subItem.attrib:
                r = int(subItem.attrib["r"].strip())

            style_new = parseStyle(subItem,style)
            style_new["stroke-width"] = style_new["stroke-width"] * scale[0]
            im = draw_circle(im,draw,cx*scale[0],cy * scale[1],r*scale[0],style_new)

        if "rect" in subItem.tag:
            x,y,width,height,rx,ry = 0,0,0,0,0,0

            if "x" in subItem.attrib:
                x = int(subItem.attrib["x"].strip())
            if "y" in subItem.attrib:
                y = int(subItem.attrib["y"].strip())
            if "width" in subItem.attrib:
                width = int(subItem.attrib["width"].strip())
            if "height" in subItem.attrib:
                height = int(subItem.attrib["height"].strip())
            if "rx" in subItem.attrib:
                rx = int(subItem.attrib["rx"].strip())
            if "ry" in subItem.attrib:
                ry = int(subItem.attrib["ry"].strip())

            style_new = parseStyle(subItem,style)
            style_new["stroke-width"] = style_new["stroke-width"] * scale[0]
            im = draw_rect(im,draw,x * scale[0] ,y * scale[1],width * scale[0],height * scale[1],
                rx * scale[0],ry * scale[1],style_new)

        if "ellipse" in subItem.tag:
            cx,cy,rx,ry = 0,0,0,0

            if "cx" in subItem.attrib:
                cx = int(subItem.attrib["cx"].strip())
            if "cy" in subItem.attrib:
                cy = int(subItem.attrib["cy"].strip())
            if "rx" in subItem.attrib:
                rx = int(subItem.attrib["rx"].strip())
            if "ry" in subItem.attrib:
                ry = int(subItem.attrib["ry"].strip())

            style_new = parseStyle(subItem,style)
            style_new["stroke-width"] = style_new["stroke-width"] * scale[0]
            im = draw_ellipse(im,draw,cx * scale[0] ,cy * scale[1],rx * scale[0],ry * scale[1],style_new)

        if "line" in subItem.tag:
            x1,x2,y1,y2 = 0,0,0,0

            if "x1" in subItem.attrib:
                x1 = int(subItem.attrib["x1"].strip())
            if "y1" in subItem.attrib:
                y1 = int(subItem.attrib["y1"].strip())
            if "x2" in subItem.attrib:
                x2 = int(subItem.attrib["x2"].strip())
            if "y2" in subItem.attrib:
                y2 = int(subItem.attrib["y2"].strip())

            style_new = parseStyle(subItem,style)
            style_new["stroke-width"] = style_new["stroke-width"] * scale[0]

            im = draw_line(im,draw,x1* scale[0],x2 * scale[0],y1 * scale[1],y2 * scale[1],style_new) 

        if "polyline" in subItem.tag:

            points = []

            if "points" in subItem.attrib:
                for item in subItem.attrib["points"].strip().split():
                    points.append((int(item.split(",")[0]) * scale[0],int(item.split(",")[1]))*scale[1])

            style_new = parseStyle(subItem,style)
            if "fill" not in subItem.attrib:
                style_new["fill"] = (0,0,0)

            style_new["stroke-width"] = style_new["stroke-width"] * scale[0]

            print(style_new)
            im = draw_polyline(im,draw,points,style_new)


    return im,draw

def main(): 
    tree = ET.parse('test.svg')
    root = tree.getroot()

    if "svg" in root.tag:
        if "viewBox" in root.attrib:
            im = Image.new('RGB', (int(root.attrib["viewBox"].split()[2]), int(root.attrib["viewBox"].split()[3])),"White")
        elif "width" in root.attrib and "height" in root.attrib:
            im = Image.new('RGB', (int(root.attrib["width"]), int(root.attrib["height"])),"White")
        else:
            print("SVG file invalid")
            exit()
    else:
        print("SVG file invalid")
        exit()

    style = parseStyle(root,{})
    draw = ImageDraw.Draw(im)
    parseSVG(im,draw,root,(1,1),style)



    





    # fill_coll = "Red"
    # fill_tr = (int(colors.to_rgb(fill_coll)[0]*255),int(colors.to_rgb(fill_coll)[1]*255),int(colors.to_rgb(fill_coll)[2]*255))

    # stroke_coll = "Red"
    # stroke_tr = (int(colors.to_rgb(stroke_coll)[0]*255),int(colors.to_rgb(stroke_coll)[1]*255),int(colors.to_rgb(stroke_coll)[2]*255))

    # style = {"fill":fill_tr,"stroke":stroke_tr,"stroke-width" : 10}
    # draw_path(im,draw, [["M",(60,100)],["A",[60,40,10,0,0,(140,100)]]], style = style)


    im.save("test.png", "PNG")

if __name__ == "__main__": 
    main() 


# fill_coll = "Red"
# fill_tr = (int(colors.to_rgb(fill_coll)[0]*255),int(colors.to_rgb(fill_coll)[1]*255),int(colors.to_rgb(fill_coll)[2]*255))

# stroke_coll = "Red"
# stroke_tr = (int(colors.to_rgb(stroke_coll)[0]*255),int(colors.to_rgb(stroke_coll)[1]*255),int(colors.to_rgb(stroke_coll)[2]*255))

# style = {"fill":fill_tr,"stroke":stroke_tr,"stroke-width" : 10}

# draw_rect(draw,xcoord= 20 , ycoord = 20, width = 300, height = 90, style = style, rx = 95, ry = 15)
# draw_circle(draw, cx = 500, cy = 500, r = 100, style = style)
# draw_ellipse(draw, cx = 1000, cy = 500, rx = 100, ry = 50, style = style)
# draw_line(draw, x1 = 10, y1 = 10, x2 = 10, y2= 500, style = style)
# draw_polyline(draw, [(20,120), (70,45), (70,95), (120,20), (200,50), (10,10), (100,150)], style = style)
# draw_path(draw, [["M",(10,90)],["C",[(30,90),(25,10),(50,10)]],["S",[(70,90),(90,90)]]], style = style)
# draw_path(draw, [["M",(10,50)],["Q",[(25,25),(40,50)]],["t",[(30,0)]],["t",[(30,0)]],["t",[(30,0)]],["t",[(30,0)]]], style = style)
# draw_path(draw, [["M",(60,100)],["A",[60,40,10,1,0,(140,100)]]], style = style)
# draw_path(draw, [["M",(60,100)],["A",[60,40,10,1,1,(140,100)]]], style = style)
# draw_path(draw, [["M",(60,100)],["A",[60,40,10,0,1,(140,100)]]], style = style)
# draw_path(draw, [["M",(60,100)],["A",[60,40,10,0,0,(140,100)]]], style = style)
# draw_circle(draw, cx = 100, cy = 100, r = 1, style = style)
# draw_path(draw, [["M",(250,10)],["l",(-40,80)],["l",(80,0)],["z"]], style = style)



