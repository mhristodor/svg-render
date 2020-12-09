import xml.etree.ElementTree as ET
import pprint

tree = ET.parse('test.svg')
root = tree.getroot()

for child in root :
	if child.tag == "rect":
		print(child.attrib)