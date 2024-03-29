from PIL import Image, ImageDraw, ImageOps
from matplotlib import colors

import xml.etree.ElementTree as ET
import pprint
import sys
import PIL
import numpy as np
import math
import io
import cairo
import argparse


def getAngle(cx, cy, x, y):
    """Returns the angle between the center of a circle and 2 points forming an arc"""

    angle = math.atan2(cx - x, cy - y) * (180 / math.pi) + 90
    if angle < 0:
        angle += 360

    return angle


def vectorAngle(ux, uy, vx, vy):
    """Returns the vector angle between 2 vectors"""

    sign = None
    if ux * vy - uy * vx < 0:
        sign = -1
    else:
        sign = 1

    ua = math.sqrt(ux * ux + uy * uy)
    va = math.sqrt(vx * vx + vy * vy)
    dot = ux * vx + uy * vy

    return sign * math.acos(dot / (ua * va))


def get_center(x1, y1, x2, y2, fa, fs, rx, ry, phi):
    """Returns center and 2 angles of a rotated ellipse"""

    sinphi = math.sin(math.radians(phi))
    cosphi = math.cos(math.radians(phi))

    x = cosphi * (x1 - x2) / 2 + sinphi * (y1 - y2) / 2
    y = -sinphi * (x1 - x2) / 2 + cosphi * (y1 - y2) / 2

    px = x * x
    py = y * y

    prx = rx * rx
    pry = ry * ry

    L = px / prx + py / pry

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

    M = math.sqrt(abs((prx * pry - prx * py - pry * px) / (prx * py
                  + pry * px))) * sign

    _cx = M * (rx * y) / ry
    _cy = M * -(ry * x) / rx

    cx = cosphi * _cx - sinphi * _cy + (x1 + x2) / 2
    cy = sinphi * _cx + cosphi * _cy + (y1 + y2) / 2

    theta = math.degrees(vectorAngle(1, 0, (x - _cx) / rx, (y - _cy)
                         / ry))

    _dTheta = math.degrees(vectorAngle((x - _cx) / rx, (y - _cy) / ry,
                           (-x - _cx) / rx, (-y - _cy) / ry)) % 360

    if fs == 0 and _dTheta > 0:
        _dTheta -= 360
    elif fs == 1 and _dTheta < 0:
        _dTheta += 360
    
    return (cx, cy, getAngle(cx, cy, x1, y1), getAngle(cx, cy, x2, y2))


def line(p1, p2):
    """Return line formulae given 2 points"""
    
    A = p1[1] - p2[1]
    B = p2[0] - p1[0]
    C = p1[0] * p2[1] - p2[0] * p1[1]
    
    return (A, B, -C)


def intersection(L1, L2):
    """Returns the intersection point of 2 lines"""
    
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    
    if D != 0:
        x = Dx / D
        y = Dy / D
        return (x, y)
    else:
        return False

def draw_rect(im, draw, xcoord=0, ycoord=0, width=0, height=0,
              rx=0, ry=0, style={'fill': None, 'stroke': 'Black', 'stroke-width': 1}):
    """Draws a rectangle on the specified area"""

    style = defaultStyle(style)

    if rx == 0 and ry != 0:
        rx = ry

    if ry == 0 and rx != 0:
        ry = rx

    if rx == ry == 0:
        draw.line([(xcoord + rx, ycoord), (xcoord + rx, ycoord
                  + height), (xcoord + width - rx, ycoord + height),
                  (xcoord + width - rx, ycoord), (xcoord + rx,
                  ycoord)], fill=style['stroke'],
                  width=style['stroke-width'], joint='curve')
    else:
        draw.line([(xcoord + rx, ycoord + height), (xcoord + width
                  - rx, ycoord + height)], fill=style['stroke'],
                  width=style['stroke-width'], joint='curve')
        draw.line([(xcoord + rx, ycoord), (xcoord - rx + width,
                  ycoord)], fill=style['stroke'],
                  width=style['stroke-width'], joint='curve')

        draw.line([(xcoord, ycoord + ry), (xcoord, ycoord + height
                  - ry)], fill=style['stroke'],
                  width=style['stroke-width'], joint='curve')
        draw.line([(xcoord + width, ycoord + ry), (xcoord + width,
                  ycoord + height - ry)], fill=style['stroke'],
                  width=style['stroke-width'], joint='curve')

        draw.arc([xcoord, ycoord, xcoord + rx * 2, ycoord + ry * 2],
                 180, 270, style['stroke'], width=style['stroke-width'])
        draw.arc([xcoord, ycoord + height - 2 * ry, xcoord + rx * 2,
                 ycoord + height], 90, 180, style['stroke'],
                 width=style['stroke-width'])
        draw.arc([xcoord + width - rx * 2, ycoord + height - ry * 2,
                 xcoord + width, ycoord + height], 0, 90, style['stroke'
                 ], width=style['stroke-width'])
        draw.arc([xcoord + width - rx * 2, ycoord, xcoord + width,
                 ycoord + ry * 2], 270, 0, style['stroke'],
                 width=style['stroke-width'])

    if style['fill'] != None:
        ImageDraw.floodfill(im, (xcoord + width / 2, ycoord + height
                            / 2), style['fill'])
    else:
        ImageDraw.floodfill(im, (xcoord + width / 2, ycoord + height
                            / 2), (0, 0, 0))

    return im


def draw_circle(im, draw, cx=0, cy=0, r=0,
                style={'fill': None, 'stroke': 'Black', 'stroke-width': 1}):
    """Draws a circle on the specified area"""
    
    style = defaultStyle(style)
    draw.arc([cx - r, cy - r, cx + r, cy + r], 0, 360, style['stroke'],
             width=style['stroke-width'])

    if style['fill'] != None:
        ImageDraw.floodfill(im, (cx, cy), style['fill'])
    else:
        ImageDraw.floodfill(im, (cx, cy), (0, 0, 0))

    return im


def draw_ellipse(im, draw, cx=0, cy=0,
                 rx=0, ry=0, style={'fill': None, 'stroke': 'Black', 'stroke-width': 1}):
    """Draws an ellipse on the specified area"""
    
    style = defaultStyle(style)

    draw.arc([cx - rx, cy - ry, cx + rx, cy + ry], 0, 360,
             style['stroke'], width=style['stroke-width'])

    if style['fill'] != None:
        ImageDraw.floodfill(im, (cx, cy), style['fill'])
    else:
        ImageDraw.floodfill(im, (cx, cy), (0, 0, 0))
    
    return im


def draw_line(im, draw, x1=0, x2=0, y1=0, y2=0,
              style={'fill': None, 'stroke': 'Black', 'stroke-width': 1}):
    """Draws a line on the specified area"""

    style = defaultStyle(style)

    draw.line([(x1, y1), (x2, y2)], fill=style['stroke'],
              width=style['stroke-width'], joint='curve')

    return im


def draw_polyline(im, draw, points=[(0, 0)],
                  style={'fill': None, 'stroke': 'Black', 'stroke-width': 1}):
    """Draws a polyline on the specified area"""

    style = defaultStyle(style)

    if style['fill'] != None:
        points.append(points[0])
        draw.line(points, fill=style['stroke'],
                  width=style['stroke-width'], joint='curve')

        noPint = True
        
        for i in range(len(points) - 2):
            for j in range(i, len(points) - 1):

                L1 = line(points[i], points[i + 1])
                L2 = line(points[j], points[j + 1])

                pint = intersection(L1, L2)
                if pint:
                    (x1, x2, x3) = (points[i][0], points[i + 1][0],
                                    pint[0])
                    (y1, y2, y3) = (points[i][1], points[i + 1][1],
                                    pint[1])

                    onSeg = min(x1, x2) <= x3 <= max(x1, x2) \
                        and min(y1, y2) <= y3 <= max(y1, y2)
                    pint = (int(pint[0]), int(pint[1]))

                    if onSeg and pint not in points:

                        noPint = False
                        
                        ImageDraw.floodfill(im, ((((points[i + 1][0]
                                + points[j][0]) / 2 + pint[0]) / 2
                                + pint[0]) / 2 + style['stroke-width']
                                / 2, (((points[i + 1][1]
                                + points[j][1]) / 2 + pint[1]) / 2
                                + pint[1]) / 2 + style['stroke-width']
                                / 2), style['fill'])
                        
                        ImageDraw.floodfill(im, ((((points[i][0]
                                + points[j + 1][0]) / 2 + pint[0]) / 2
                                + pint[0]) / 2 - style['stroke-width']
                                / 2, (((points[i][1] + points[j
                                + 1][1]) / 2 + pint[1]) / 2 + pint[1])
                                / 2 - style['stroke-width'] / 2),
                                style['fill'])

        if noPint:
            ImageDraw.floodfill(im, ((points[1][0] + points[2][0]) / 2 
                                - style["stroke-width"], (points[1][1]
                                + points[2][1]) / 2 - style["stroke-width"]),
                                style['fill'])
    else:

        draw.line(points, fill=style['stroke'],
                  width=style['stroke-width'], joint='curve')

    return im


def draw_path(im, draw, descr=None,
              style={'fill': None, 'stroke': 'Black', 'stroke-width': 1}):
    """Draws a path on the specified area"""

    style = defaultStyle(style)

    startPoint = None
    startPointQuad = None
    lastCmd = None
    lastCmdQuad = None
    initialPoint = None
    pointsMulty = []
    flagZ = False

    if descr != None:
        for cmd in descr:
            if cmd[0] == 'M':
                startPoint = cmd[1]
                lastCmd = None
                lastCmdQuad = None

                if initialPoint == None:
                    initialPoint = cmd[1]

                pointsMulty.append([cmd[1]])
                continue
            
            if cmd[0] == 'm':
                startPoint = (startPoint[0] + cmd[1][0], startPoint[1]
                              + cmd[1][1])
                lastCmd = None
                lastCmdQuad = None

                pointsMulty[-1].append(startPoint)

                continue
            
            if cmd[0] == 'L':
                draw.line([startPoint, cmd[1]], fill=style['stroke'],
                          width=style['stroke-width'], joint='curve')
                startPoint = cmd[1]
                lastCmd = None
                lastCmdQuad = None

                pointsMulty[-1].append(startPoint)
                continue
            
            if cmd[0] == 'l':
                draw.line([startPoint, (startPoint[0] + cmd[1][0],
                          startPoint[1] + cmd[1][1])],
                          fill=style['stroke'],
                          width=style['stroke-width'], joint='curve')
                startPoint = (startPoint[0] + cmd[1][0], startPoint[1]
                              + cmd[1][1])
                lastCmdQuad = None
                lastCmd = None

                pointsMulty[-1].append(startPoint)

                continue
            
            if cmd[0] == 'H':
                draw.line([startPoint, (cmd[1], startPoint[1])],
                          fill=style['stroke'],
                          width=style['stroke-width'], joint='curve')
                startPoint = (cmd[1], startPoint[1])
                lastCmdQuad = None
                lastCmd = None

                pointsMulty[-1].append(startPoint)
                continue
            
            if cmd[0] == 'h':
                draw.line([startPoint, (startPoint[0] + cmd[1],
                          startPoint[1])], fill=style['stroke'],
                          width=style['stroke-width'], joint='curve')
                startPoint = (startPoint[0] + cmd[1], startPoint[1])
                lastCmdQuad = None
                lastCmd = None
                pointsMulty[-1].append(startPoint)
                continue
            
            if cmd[0] == 'V':
                draw.line([startPoint, (startPoint[0], cmd[1])],
                          fill=style['stroke'],
                          width=style['stroke-width'], joint='curve')
                startPoint = (startPoint[0], cmd[1])
                lastCmdQuad = None
                lastCmd = None
                pointsMulty[-1].append(startPoint)
                continue
            
            if cmd[0] == 'v':
                draw.line([startPoint, (startPoint[0], startPoint[1]
                          + cmd[1])], fill=style['stroke'],
                          width=style['stroke-width'], joint='curve')
                startPoint = (startPoint[0], startPoint[1] + cmd[1])
                lastCmdQuad = None
                lastCmd = None
                pointsMulty[-1].append(startPoint)
                continue
            
            if cmd[0] == 'C':

                surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                        im.size[0], im.size[1])
                ctx = cairo.Context(surface)
                ctx.set_source_rgba(0.0, 0.0, 0.0, 0.0)
                ctx.paint()
                ctx.move_to(startPoint[0], startPoint[1])
                ctx.curve_to(cmd[1][0][0],
                             cmd[1][0][1],
                             cmd[1][1][0],
                             cmd[1][1][1],
                             cmd[1][2][0],
                             cmd[1][2][1])
                ctx.set_source_rgb(style['stroke'][0], 
                                   style['stroke'][1], 
                                   style['stroke'][2])
                ctx.set_line_width(style['stroke-width'])
                ctx.stroke()

                buf = io.BytesIO()
                surface.write_to_png(buf)
                buf.seek(0)

                im_tmp = Image.open(buf)

                offset = (0, 0)

                im.paste(im_tmp, offset, mask=im_tmp)

                lastCmd = (2 * cmd[1][2][0] - cmd[1][1][0], 
                           2 * cmd[1][2][1] - cmd[1][1][1])
                startPoint = (cmd[1][2][0], cmd[1][2][1])
                lastCmdQuad = None

                pointsMulty[-1].append(startPoint)

                continue
            
            if cmd[0] == 'c':

                surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                        im.size[0], im.size[1])
                ctx = cairo.Context(surface)
                ctx.set_source_rgba(0.0, 0.0, 0.0, 0.0)
                ctx.paint()
                ctx.move_to(startPoint[0], startPoint[1])
                ctx.curve_to(startPoint[0] + cmd[1][0][0],
                             startPoint[1] + cmd[1][0][1],
                             startPoint[0] + cmd[1][1][0],
                             startPoint[1] + cmd[1][1][1],
                             startPoint[0] + cmd[1][2][0],
                             startPoint[1] + cmd[1][2][1])
                ctx.set_source_rgb(style['stroke'][0], style['stroke'
                                   ][1], style['stroke'][2])
                ctx.set_line_width(style['stroke-width'])
                ctx.stroke()

                buf = io.BytesIO()
                surface.write_to_png(buf)
                buf.seek(0)

                im_tmp = Image.open(buf)

                offset = (0, 0)

                im.paste(im_tmp, offset, mask=im_tmp)

                lastCmd = (2 * (startPoint[0] + cmd[1][2][0])
                           - startPoint[0] - cmd[1][1][0], 2
                           * (startPoint[1] + cmd[1][2][1])
                           - startPoint[1] - cmd[1][1][1])
                startPoint = (startPoint[0] + cmd[1][2][0],
                              startPoint[1] + cmd[1][2][1])
                lastCmdQuad = None

                pointsMulty[-1].append(startPoint)

                continue
            
            if cmd[0] == 'S':

                surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                        im.size[0], im.size[1])
                ctx = cairo.Context(surface)
                ctx.set_source_rgba(0.0, 0.0, 0.0, 0.0)
                ctx.paint()
                ctx.move_to(startPoint[0], startPoint[1])

                if lastCmd != None:
                    ctx.curve_to(lastCmd[0],
                                 lastCmd[1],
                                 cmd[1][0][0],
                                 cmd[1][0][1],
                                 cmd[1][1][0],
                                 cmd[1][1][1])
                else:
                    ctx.curve_to(startPoint[0],
                                 startPoint[1],
                                 cmd[1][0][0],
                                 cmd[1][0][1],
                                 cmd[1][1][0],
                                 cmd[1][1][1])

                ctx.set_source_rgb(style['stroke'][0], 
                                   style['stroke'][1], 
                                   style['stroke'][2])
                ctx.set_line_width(style['stroke-width'])
                ctx.stroke()

                buf = io.BytesIO()
                surface.write_to_png(buf)
                buf.seek(0)

                im_tmp = Image.open(buf)

                offset = (0, 0)

                im.paste(im_tmp, offset, mask=im_tmp)

                lastCmd = (2 * cmd[1][1][0] - cmd[1][0][0], 
                           2 * cmd[1][1][1] - cmd[1][0][1])
                startPoint = (cmd[1][1][0], cmd[1][1][1])
                lastCmdQuad = None

                pointsMulty[-1].append(startPoint)

                continue
            
            if cmd[0] == 's':

                surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                        im.size[0], im.size[1])
                ctx = cairo.Context(surface)
                ctx.set_source_rgba(0.0, 0.0, 0.0, 0.0)
                ctx.paint()
                ctx.move_to(startPoint[0], startPoint[1])

                if lastCmd != None:
                    ctx.curve_to(lastCmd[0],
                                 lastCmd[1],
                                 startPoint[0] + cmd[1][0][0],
                                 startPoint[1] + cmd[1][0][1],
                                 startPoint[0] + cmd[1][1][0],
                                 startPoint[1] + cmd[1][1][1])
                else:
                    ctx.curve_to(startPoint[0],
                                 startPoint[1],
                                 startPoint[0] + cmd[1][0][0],
                                 startPoint[1] + cmd[1][0][1],
                                 startPoint[0] + cmd[1][1][0],
                                 startPoint[1] + cmd[1][1][1])

                ctx.set_source_rgb(style['stroke'][0], 
                                   style['stroke'][1], 
                                   style['stroke'][2])
                ctx.set_line_width(style['stroke-width'])
                ctx.stroke()

                buf = io.BytesIO()
                surface.write_to_png(buf)
                buf.seek(0)

                im_tmp = Image.open(buf)

                offset = (0, 0)

                im.paste(im_tmp, offset, mask=im_tmp)

                lastCmd = (2 * (startPoint[0] + cmd[1][1][0])
                           - startPoint[0] + cmd[1][0][0], 
                           2 * (startPoint[1] + cmd[1][1][1])
                           - startPoint[1] + cmd[1][0][1])
                startPoint = (startPoint[0] + cmd[1][1][0],
                              startPoint[1] + cmd[1][1][1])
                lastCmdQuad = None

                pointsMulty[-1].append(startPoint)

                continue
            
            if cmd[0] == 'Q':

                surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                        im.size[0], im.size[1])
                ctx = cairo.Context(surface)
                ctx.set_source_rgba(0.0, 0.0, 0.0, 0.0)
                ctx.paint()
                ctx.move_to(startPoint[0], startPoint[1])
                ctx.curve_to(2 / 3 * cmd[1][0][0] + 1 / 3 * startPoint[0],
                             2 / 3 * cmd[1][0][1] + 1 / 3 * startPoint[1],
                             2 / 3 * cmd[1][0][0] + 1 / 3 * cmd[1][1][0],
                             2 / 3 * cmd[1][0][1] + 1 / 3 * cmd[1][1][1],
                             cmd[1][1][0],
                             cmd[1][1][1])
                ctx.set_source_rgb(style['stroke'][0], style['stroke'
                                   ][1], style['stroke'][2])
                ctx.set_line_width(style['stroke-width'])
                ctx.stroke()

                buf = io.BytesIO()
                surface.write_to_png(buf)
                buf.seek(0)

                im_tmp = Image.open(buf)

                offset = (0, 0)

                im.paste(im_tmp, offset, mask=im_tmp)

                lastCmd = None
                startPoint = (cmd[1][1][0], cmd[1][1][1])
                lastCmdQuad = (2 * cmd[1][1][0] - cmd[1][0][0], 
                               2 * cmd[1][1][1] - cmd[1][0][1])

                pointsMulty[-1].append(startPoint)

                continue

            if cmd[0] == 'q':

                surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                        im.size[0], im.size[1])
                ctx = cairo.Context(surface)
                ctx.set_source_rgba(0.0, 0.0, 0.0, 0.0)
                ctx.paint()
                ctx.move_to(startPoint[0], startPoint[1])
                ctx.curve_to(2 / 3 * (startPoint[0] + cmd[1][0][0]) + 1 / 3
                             * startPoint[0],
                             2 / 3 * (startPoint[1] + cmd[1][0][1]) + 1 / 3
                             * startPoint[1],
                             2 / 3 * (startPoint[0] + cmd[1][0][0]) + 1 / 3
                             * (startPoint[0] + cmd[1][1][0]),
                             2 / 3 * (startPoint[1] + cmd[1][0][1]) + 1 / 3
                             * (startPoint[1] + cmd[1][1][1]),
                             startPoint[0] + cmd[1][1][0],
                             startPoint[1] + cmd[1][1][1])
                ctx.set_source_rgb(style['stroke'][0], 
                                   style['stroke'][1], 
                                   style['stroke'][2])
                ctx.set_line_width(style['stroke-width'])
                ctx.stroke()

                buf = io.BytesIO()
                surface.write_to_png(buf)
                buf.seek(0)

                im_tmp = Image.open(buf)

                offset = (0, 0)

                im.paste(im_tmp, offset, mask=im_tmp)

                lastCmd = None
                startPoint = (startPoint[0] + cmd[1][1][0],
                              startPoint[1] + cmd[1][1][1])
                lastCmdQuad = (2 * (startPoint[0] + cmd[1][1][0])
                               - cmd[1][0][0] - startPoint[0], 
                               2 * (cmd[1][1][1] + startPoint[1])
                               - cmd[1][0][1] - startPoint[1])

                pointsMulty[-1].append(startPoint)

                continue

            if cmd[0] == 'T':
                nextPoint = None

                surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                        im.size[0], im.size[1])
                ctx = cairo.Context(surface)
                ctx.set_source_rgba(0.0, 0.0, 0.0, 0.0)
                ctx.paint()
                ctx.move_to(startPoint[0], startPoint[1])
                if lastCmdQuad != None:
                    ctx.curve_to(2 / 3 * lastCmdQuad[0] + 1 / 3 * startPoint[0],
                                 2 / 3 * lastCmdQuad[1] + 1 / 3 * startPoint[1],
                                 2 / 3 * lastCmdQuad[0] + 1 / 3 * cmd[1][0][0],
                                 2 / 3 * lastCmdQuad[1] + 1 / 3 * cmd[1][0][1],
                                 cmd[1][0][0],
                                 cmd[1][0][1])
                    nextPoint = [lastCmdQuad[0], lastCmdQuad[1]]
                else:

                    ctx.curve_to(2 / 3 * startPoint[0] + 1 / 3 * startPoint[0],
                                 2 / 3 * startPoint[1] + 1 / 3 * startPoint[1],
                                 2 / 3 * startPoint[0] + 1 / 3 * cmd[1][0][0],
                                 2 / 3 * startPoint[1] + 1 / 3 * cmd[1][0][1],
                                 cmd[1][0][0],
                                 cmd[1][0][1])
                    nextPoint = [startPoint[0], startPoint[1]]

                ctx.set_source_rgb(style['stroke'][0], 
                                   style['stroke'][1], 
                                   style['stroke'][2])
                ctx.set_line_width(style['stroke-width'])
                ctx.stroke()

                buf = io.BytesIO()
                surface.write_to_png(buf)
                buf.seek(0)

                im_tmp = Image.open(buf)

                offset = (0, 0)

                im.paste(im_tmp, offset, mask=im_tmp)

                lastCmd = None
                startPoint = (cmd[1][0][0], cmd[1][0][1])
                lastCmdQuad = (2 * cmd[1][0][0] - nextPoint[0], 
                               2 * cmd[1][0][1] - nextPoint[1])

                pointsMulty[-1].append(startPoint)

                continue

            if cmd[0] == 't':
                nextPoint = None

                surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                        im.size[0], im.size[1])
                ctx = cairo.Context(surface)
                ctx.set_source_rgba(0.0, 0.0, 0.0, 0.0)
                ctx.paint()
                ctx.move_to(startPoint[0], startPoint[1])
                if lastCmdQuad != None:
                    ctx.curve_to(2 / 3 * lastCmdQuad[0] + 1 / 3 * startPoint[0],
                                 2 / 3 * lastCmdQuad[1] + 1 / 3 * startPoint[1],
                                 2 / 3 * lastCmdQuad[0] + 1 / 3 * (startPoint[0]
                                 + cmd[1][0][0]),
                                 2 / 3 * lastCmdQuad[1] + 1 / 3 * (startPoint[1]
                                 + cmd[1][0][1]),
                                 startPoint[0] + cmd[1][0][0],
                                 startPoint[1] + cmd[1][0][1])
                    nextPoint = [lastCmdQuad[0], lastCmdQuad[1]]
                else:

                    ctx.curve_to(2 / 3 * startPoint[0] + 1 / 3 * startPoint[0],
                                 2 / 3 * startPoint[1] + 1 / 3 * startPoint[1],
                                 2 / 3 * startPoint[0] + 1 / 3 * (startPoint[0]
                                 + cmd[1][0][0]),
                                 2 / 3 * startPoint[1] + 1 / 3 * (startPoint[1]
                                 + cmd[1][0][1]),
                                 startPoint[0] + cmd[1][0][0],
                                 startPoint[1] + cmd[1][0][1])
                    nextPoint = [startPoint[0], startPoint[1]]

                ctx.set_source_rgb(style['stroke'][0], style['stroke'
                                   ][1], style['stroke'][2])
                ctx.set_line_width(style['stroke-width'])
                ctx.stroke()

                buf = io.BytesIO()
                surface.write_to_png(buf)
                buf.seek(0)

                im_tmp = Image.open(buf)

                offset = (0, 0)

                im.paste(im_tmp, offset, mask=im_tmp)

                lastCmd = None
                startPoint = (startPoint[0] + cmd[1][0][0],
                              startPoint[1] + cmd[1][0][1])
                lastCmdQuad = (2 * startPoint[0] - nextPoint[0], 
                               2 * startPoint[1] - nextPoint[1])

                pointsMulty[-1].append(startPoint)
                continue

            if cmd[0] == 'A':
                (cx, cy, p1, p2) = get_center(startPoint[0], startPoint[1],
                                              cmd[1][5][0], cmd[1][5][1], 
                                              cmd[1][3], cmd[1][4], 
                                              cmd[1][0], cmd[1][1], cmd[1][2])

                im2 = Image.new('RGBA', (2 * cmd[1][0]
                                + style['stroke-width'], 2 * cmd[1][1]
                                + style['stroke-width']))
                draw2 = ImageDraw.Draw(im2)

                if cmd[1][3] == 1 and cmd[1][4] == 1 or cmd[1][3] == 0 \
                    and cmd[1][4] == 1:
                    draw2.arc([0, 0, 2 * cmd[1][0]
                              + style['stroke-width'], 2 * cmd[1][1]
                              + style['stroke-width']], p2 + cmd[1][2],
                              p1 + cmd[1][2], style['stroke'],
                              width=style['stroke-width'])
                else:
                    draw2.arc([0, 0, 2 * cmd[1][0]
                              + style['stroke-width'], 2 * cmd[1][1]
                              + style['stroke-width']], p1 + cmd[1][2],
                              p2 + cmd[1][2], style['stroke'],
                              width=style['stroke-width'])

                im_new = im2.rotate(cmd[1][2])
                im_flip = ImageOps.flip(im_new)

                offset = (int(cx - cmd[1][0] - style['stroke-width'])
                          + 1, int(cy - cmd[1][1] - style['stroke-width'
                          ]) + 1)

                im.paste(im_flip, offset, mask=im_flip)

                startPoint = cmd[1][5]

                pointsMulty[-1].append(startPoint)
                continue

            if cmd[0] == 'a':
                (cx, cy, p1, p2) = get_center(startPoint[0], startPoint[1],
                                              cmd[1][5][0] + startPoint[0],
                                              cmd[1][5][1] + startPoint[1],
                                              cmd[1][3], cmd[1][4],
                                              cmd[1][0], cmd[1][1], cmd[1][2])

                im2 = Image.new('RGBA', (int(cx) + 2 * cmd[1][0],
                                int(cy) + 2 * cmd[1][1]))
                draw2 = ImageDraw.Draw(im2)

                if cmd[1][3] == 1 and cmd[1][4] == 1 or cmd[1][3] == 0 \
                    and cmd[1][4] == 1:
                    draw2.arc([0, 0, 2 * cmd[1][0]
                              + style['stroke-width'], 2 * cmd[1][1]
                              + style['stroke-width']], p2 + cmd[1][2],
                              p1 + cmd[1][2], style['stroke'],
                              width=style['stroke-width'])
                else:
                    draw2.arc([0, 0, 2 * cmd[1][0]
                              + style['stroke-width'], 2 * cmd[1][1]
                              + style['stroke-width']], p1 + cmd[1][2],
                              p2 + cmd[1][2], style['stroke'],
                              width=style['stroke-width'])
                
                im_new = im2.rotate(cmd[1][2])
                im_flip = ImageOps.flip(im_new)
                
                offset = (int(cx - cmd[1][0] - style['stroke-width'])
                          + 1, int(cy - cmd[1][1] - style['stroke-width'
                          ]) + 1)

                im.paste(im_flip, offset, mask=im_flip)

                startPoint = (cmd[1][5][0] + startPoint[0],
                              cmd[1][5][1] + startPoint[1])
                pointsMulty[-1].append(startPoint)
                continue

            if cmd[0].lower() == 'z':
                draw_line(im,draw, initialPoint[0], startPoint[0], 
                          initialPoint[1], startPoint[1], style=style)
                startPoint = None
                initialPoint = None
                flagZ = True
                continue

    if style['fill'] != None and not flagZ:
        draw_line(im,draw, initialPoint[0], startPoint[0], 
                          initialPoint[1], startPoint[1],
                          style={'fill': None, 'stroke': style['fill'], 
                          'stroke-width': 1})

    for points in pointsMulty:
        if style['fill'] != None:
            points.append(points[0])
            noPint = True
            for i in range(len(points) - 2):
                for j in range(i, len(points) - 1):

                    L1 = line(points[i], points[i + 1])
                    L2 = line(points[j], points[j + 1])

                    pint = intersection(L1, L2)
                    if pint:
                        (x1, x2, x3) = (points[i][0], points[i + 1][0],
                                pint[0])
                        (y1, y2, y3) = (points[i][1], points[i + 1][1],
                                pint[1])

                        onSeg = min(x1, x2) <= x3 <= max(x1, x2) \
                            and min(y1, y2) <= y3 <= max(y1, y2)
                        pint = (int(pint[0]), int(pint[1]))

                        if onSeg and pint not in points:

                            noPint = False
                            ImageDraw.floodfill(im, ((((points[i
                                    + 1][0] + points[j][0]) / 2
                                    + pint[0]) / 2 + pint[0]) / 2
                                    + style['stroke-width'] / 2,
                                    (((points[i + 1][1] + points[j][1])
                                    / 2 + pint[1]) / 2 + pint[1]) / 2
                                    + style['stroke-width'] / 2),
                                    style['fill'])
                            ImageDraw.floodfill(im, ((((points[i][0]
                                    + points[j + 1][0]) / 2 + pint[0])
                                    / 2 + pint[0]) / 2
                                    - style['stroke-width'] / 2,
                                    (((points[i][1] + points[j + 1][1])
                                    / 2 + pint[1]) / 2 + pint[1]) / 2
                                    - style['stroke-width'] / 2),
                                    style['fill'])

            if noPint:
                ImageDraw.floodfill(im, ((points[1][0] + points[2][0]) / 2 
                                    - style["stroke-width"], (points[1][1]
                                    + points[2][1]) / 2 - style["stroke-width"]),
                                     style['fill'])
    return im


def parseStyle(subItem, style):
    """Returns style dictionary after parsing the input"""

    style_new = style

    if 'stroke' in subItem.attrib:
        stroke_coll = subItem.attrib['stroke'].strip()
        stroke_tr = (int(colors.to_rgb(stroke_coll)[0] * 255),
                     int(colors.to_rgb(stroke_coll)[1] * 255),
                     int(colors.to_rgb(stroke_coll)[2] * 255))
        style_new['stroke'] = stroke_tr

    if 'fill' in subItem.attrib:
        if subItem.attrib['fill'].lower() == 'none':
            style_new['fill'] = None
        else:
            fill_coll = subItem.attrib['fill'].strip()
            fill_tr = (int(colors.to_rgb(fill_coll)[0] * 255),
                       int(colors.to_rgb(fill_coll)[1] * 255),
                       int(colors.to_rgb(fill_coll)[2] * 255))
            style_new['fill'] = fill_tr

    if 'stroke-width' in subItem.attrib:
        stroke_width = int(subItem.attrib['stroke-width'].strip())
        style_new['stroke-width'] = stroke_width

    return style_new


def defaultStyle(style):
    """Returns style dictionary with default values"""

    if 'fill' not in style.keys():
        style['fill'] = (0, 0, 0)

    if 'stroke' not in style.keys():
        style['stroke'] = (0, 0, 0)

    if 'stroke-width' not in style.keys():
        style['stroke-width'] = 1

    return style


def representsInt(s):
    """Checks if a string represents an integer"""

    try:
        int(s)
        return True
    except ValueError:
        return False


def chunks(lst, n):
    """Returns list separated in n length chunks"""

    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def translateCommands(cmdlist):
    """Returns parsed input of Path tag d"""

    output = []
    seperateCmds = []
    tmp = []

    for item in cmdlist:
        if not representsInt(item):

            if tmp != []:
                seperateCmds.append(tmp)
            tmp = []
            tmp.append(item)

        if representsInt(item):
            tmp.append(item)

    seperateCmds.append(tmp)

    for item in seperateCmds:
        if item[0].lower() == 'm':
            tmp = chunks(item[1:], 2)
            cmd = [item[0]]
            padding = []
            for it in tmp:
                if cmd == [item[0]]:
                    cmd.append((int(it[0]), int(it[1])))
                else:
                    if item[0] == 'M':
                        padding.append(['L', (int(it[0]), int(it[1]))])
                    else:
                        padding.append(['l', (int(it[0]), int(it[1]))])
            output.append(cmd)
            if padding != []:
                output += padding

        if item[0].lower() == 'l':
            tmp = chunks(item[1:], 2)
            cmd = [item[0]]
            padding = []
            for it in tmp:
                if cmd == [item[0]]:
                    cmd.append((int(it[0]), int(it[1])))
                else:
                    padding.append([item[0], (int(it[0]), int(it[1]))])
            output.append(cmd)
            if padding != []:
                output += padding

        if item[0].lower() == 'h' or item[0].lower() == 'v':
            tmp = chunks(item[1:], 1)
            cmd = [item[0]]
            padding = []
            for it in tmp:
                if cmd == [item[0]]:
                    cmd.append(int(it[0]))
                else:
                    padding.append([item[0], int(it[0])])
            output.append(cmd)
            if padding != []:
                output += padding

        if item[0].lower() == 'c':
            tmp = chunks(item[1:], 6)
            cmd = [item[0]]
            padding = []
            for it in tmp:
                if cmd == [item[0]]:
                    cmd.append([(int(it[0]), int(it[1])), (int(it[2]),
                               int(it[3])), (int(it[4]), int(it[5]))])
                else:
                    padding.append([item[0], [(int(it[0]), int(it[1])),
                                   (int(it[2]), int(it[3])),
                                   (int(it[4]), int(it[5]))]])
            output.append(cmd)
            if padding != []:
                output += padding

        if item[0].lower() == 's':
            tmp = chunks(item[1:], 4)
            cmd = [item[0]]
            padding = []
            for it in tmp:
                if cmd == [item[0]]:
                    cmd.append([(int(it[0]), int(it[1])), (int(it[2]),
                               int(it[3]))])
                else:
                    padding.append([item[0], [(int(it[0]), int(it[1])),
                                   (int(it[2]), int(it[3]))]])
            output.append(cmd)
            if padding != []:
                output += padding

        if item[0].lower() == 'q':
            tmp = chunks(item[1:], 4)
            cmd = [item[0]]
            padding = []
            for it in tmp:
                if cmd == [item[0]]:
                    cmd.append([(int(it[0]), int(it[1])), (int(it[2]),
                               int(it[3]))])
                else:
                    padding.append([item[0], [(int(it[0]), int(it[1])),
                                   (int(it[2]), int(it[3]))]])
            output.append(cmd)
            if padding != []:
                output += padding

        if item[0].lower() == 't':
            tmp = chunks(item[1:], 2)
            cmd = [item[0]]
            padding = []
            for it in tmp:
                if cmd == [item[0]]:
                    cmd.append([(int(it[0]), int(it[1]))])
                else:
                    padding.append([item[0], [(int(it[0]),
                                   int(it[1]))]])
            output.append(cmd)
            if padding != []:
                output += padding

        if item[0].lower() == 'a':
            tmp = chunks(item[1:], 7)
            cmd = [item[0]]
            padding = []
            for it in tmp:
                if cmd == [item[0]]:
                    cmd.append([int(it[0]), int(it[1]), int(it[2]),
                                int(it[3]), int(it[4]), (int(it[5]), int(it[6]))])
                else:
                    padding.append([item[0], [int(it[0]), int(it[1]), int(it[2]),
                                   int(it[3]), int(it[4]), (int(it[5]), int(it[6]))]])
            output.append(cmd)
            if padding != []:
                output += padding

        if item[0].lower() == 'z':
            output.append(["z"])

    print(output)
    return output


def parseSVG(im, draw, root, scale,
             style={'fill': None, 'stroke': 'Black', 'stroke-width': 1}):
    """Recursive function creating the SVG and sub SVG items"""

    style = defaultStyle(style)
    try:
        for subItem in root:
            if subItem.tag.endswith('svg'):

                offsetX, offsetY, tranX, tranY, sizeX, sizeY = None, None, None, None, None, None

                if 'viewBox' in subItem.attrib:
                    sizeX = int(subItem.attrib['viewBox'].split()[2]) \
                        - int(subItem.attrib['viewBox'].split()[0])
                    sizeY = int(subItem.attrib['viewBox'].split()[3]) \
                        - int(subItem.attrib['viewBox'].split()[1])
                else:
                    (sizeX, sizeY) = im.size

                if 'height' in subItem.attrib:
                    tranY = int(subItem.attrib['height'].strip())
                else:
                    tranY = im.size[1]

                if 'width' in subItem.attrib:
                    tranX = int(subItem.attrib['width'].strip())
                else:
                    tranX = im.size[0]

                if 'x' in subItem.attrib:
                    offsetX = int(subItem.attrib['x'].strip())
                else:
                    offsetX = 0

                if 'y' in subItem.attrib:
                    offsetY = int(subItem.attrib['y'].strip())
                else:
                    offsetY = 0

                style_new = parseStyle(subItem, style)
                scale_new = (int(tranX / sizeX), int(tranY / sizeY))
                im_new = Image.new('RGB', (sizeX * scale_new[0], sizeY
                                   * scale_new[1]), 'White')
                draw_new = ImageDraw.Draw(im_new)
                style_new['stroke-width'] = style_new['stroke-width'] \
                    * scale_new[0]
                (im_new, draw_new) = parseSVG(im_new, draw_new,
                        subItem, scale_new, style_new)

                im.paste(im_new, (offsetX, offsetY))

            if 'circle' in subItem.tag:
                cx, cy, r = 0, 0, 0

                if 'cx' in subItem.attrib:
                    cx = int(subItem.attrib['cx'].strip())
                if 'cy' in subItem.attrib:
                    cy = int(subItem.attrib['cy'].strip())
                if 'r' in subItem.attrib:
                    r = int(subItem.attrib['r'].strip())

                style_new = parseStyle(subItem, style)
                style_new['stroke-width'] = style_new['stroke-width'] \
                    * scale[0]
                
                im = draw_circle(im, draw, cx * scale[0],
                                 cy * scale[1], r * scale[0], style_new)

            if 'rect' in subItem.tag:
                
                x, y, width, height, rx, ry = 0, 0, 0, 0, 0, 0

                if 'x' in subItem.attrib:
                    x = int(subItem.attrib['x'].strip())
                if 'y' in subItem.attrib:
                    y = int(subItem.attrib['y'].strip())
                if 'width' in subItem.attrib:
                    width = int(subItem.attrib['width'].strip())
                if 'height' in subItem.attrib:
                    height = int(subItem.attrib['height'].strip())
                if 'rx' in subItem.attrib:
                    rx = int(subItem.attrib['rx'].strip())
                if 'ry' in subItem.attrib:
                    ry = int(subItem.attrib['ry'].strip())

                style_new = parseStyle(subItem, style)
                style_new['stroke-width'] = style_new['stroke-width'] \
                    * scale[0]
                im = draw_rect(im, draw, x * scale[0], y * scale[1], 
                    width * scale[0], height * scale[1], rx * scale[0],
                    ry * scale[1], style_new)

            if 'ellipse' in subItem.tag:
                cx, cy, rx, ry = 0, 0, 0, 0

                if 'cx' in subItem.attrib:
                    cx = int(subItem.attrib['cx'].strip())
                if 'cy' in subItem.attrib:
                    cy = int(subItem.attrib['cy'].strip())
                if 'rx' in subItem.attrib:
                    rx = int(subItem.attrib['rx'].strip())
                if 'ry' in subItem.attrib:
                    ry = int(subItem.attrib['ry'].strip())

                style_new = parseStyle(subItem, style)
                style_new['stroke-width'] = style_new['stroke-width'] \
                    * scale[0]
                im = draw_ellipse(im, draw, cx * scale[0], cy * scale[1], 
                                  rx * scale[0], ry * scale[1], style_new)

            if 'line' in subItem.tag:
                x1, x2, y1, y2 = 0, 0, 0, 0

                if 'x1' in subItem.attrib:
                    x1 = int(subItem.attrib['x1'].strip())
                if 'y1' in subItem.attrib:
                    y1 = int(subItem.attrib['y1'].strip())
                if 'x2' in subItem.attrib:
                    x2 = int(subItem.attrib['x2'].strip())
                if 'y2' in subItem.attrib:
                    y2 = int(subItem.attrib['y2'].strip())

                style_new = parseStyle(subItem, style)
                style_new['stroke-width'] = style_new['stroke-width'] \
                    * scale[0]

                im = draw_line(im, draw, x1 * scale[0], x2 * scale[0],
                               y1 * scale[1], y2 * scale[1], style_new)

            if 'polyline' in subItem.tag:

                points = []

                if 'points' in subItem.attrib:
                    for item in subItem.attrib['points'
                            ].strip().split():
                        points.append((int(item.split(',')[0])
                                * scale[0], int(item.split(',')[1]))
                                * scale[1])

                style_new = parseStyle(subItem, style)
                if 'fill' not in subItem.attrib:
                    style_new['fill'] = (0, 0, 0)

                style_new['stroke-width'] = style_new['stroke-width'] \
                    * scale[0]

                im = draw_polyline(im, draw, points, style_new)

            if 'path' in subItem.tag:

                style_new = parseStyle(subItem, style)
                if 'fill' not in subItem.attrib:
                    style_new['fill'] = (0, 0, 0)

                style_new['stroke-width'] = style_new['stroke-width'] \
                    * scale[0]

                if 'd' in subItem.attrib:

                    fullstr = subItem.attrib['d'].strip()

                    fullstr = fullstr.replace('\n', '')
                    fullstr = fullstr.replace(',', ' ')
                    fullstr = fullstr.replace('-', ' -')
                    fullstr = fullstr.split()
                    cmd = translateCommands(fullstr)

                    im = draw_path(im, draw, cmd, style_new)
    except:
        print('Parsing error! Only int numbers accepted')
        exit()

    return (im, draw)


def main():
    """Main function"""

    parser = \
        argparse.ArgumentParser(description='SVG renderer by Hristodor Minu-Mihail')
    parser.add_argument('file_path', metavar='file', type=str,
                        help='SVG file to be rendered')
    args = parser.parse_args()

    try:
        tree = ET.parse(args.file_path)
    except:
        print('No file such as ' + args.file_path)
        exit()

    root = tree.getroot()
    if 'svg' in root.tag:
        if 'viewBox' in root.attrib:
            im = Image.new('RGB', (int(root.attrib['viewBox'].split()[2]), 
                           int(root.attrib['viewBox'].split()[3])), 'White')
        elif 'width' in root.attrib and 'height' in root.attrib:
            im = Image.new('RGB', (int(root.attrib['width']),
                           int(root.attrib['height'])), 'White')
        else:
            print('SVG file invalid')
            exit()
    else:
        print('SVG file invalid')
        exit()

    style = parseStyle(root, {})
    draw = ImageDraw.Draw(im)
    parseSVG(im, draw, root, (1, 1), style)

    im.save(args.file_path[:-3] + 'png', 'PNG')
    print('Done')
    im.show()


if __name__ == '__main__':
    main()