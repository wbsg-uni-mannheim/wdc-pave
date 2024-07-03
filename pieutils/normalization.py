import json
from tqdm import tqdm
import pandas as pd
import re
import string
from collections import defaultdict
import fractions
import os

equipment_dict_computers = {
    "Storage Solutions": [
        "HDD (Hard Disk Drive)",
        "SSD (Solid State Drive)",
        "Solid State Hard Drive",
        "Drive Storage Enclosure",
        "Tower",
        "Drive",
        "Drive Library Upgrade Option",
        "Drive Option",
        "Storage Controller (RAID)",
        "Array Controller",
        "Smart Array Controller Module",
        "RAID Controller",
        "External Cache Battery (ECB) Module",
        "Modular Array Director",
        "SAS Controller",
        "Channel SCSI Controller",
        "BBWC Upgrade",
        "Hard Drives",
        "Hard Drive",
        "SSD (Solid Sate Drive)",
        "Modular Smart Array SAN Director",
        "RAID Storage Controller",
        "Smart Array",
        "Raid Controller",
        "Multi-Server Card",
        "Battery Backup",
        "Memory Cartridge",
        "HardDrive",
        "Secure Digital",
        "EMUBOARD MODULAR ARRAY",
        "Server Blade",
        "Rackmount Drive Library",
        "CTO Blade",
        "Storage System",
        "SDRAM Memory",
        "EDO Memory",
        "Drv",
        "HDD",
        "SSD",
        "RAID Controller Kit",
        "Solid State Drive",
        "Midline Drive",
        "ClusterStorage",
        "enclosure",
        "Rackmount LTO Tape Library AutoLoader",
        "Smart Array Cluster Storage",
        "Entry SAN",
        "Solid State Drive (SSD)",
        "Internal Array Controller Cabling Option Kit",
        "Flash Card",
        "Desktop Hard Drive",
        "RAID Array",
        "CD-RW/DVD-ROM Combo Drive",
        "HD",
        "Array Cabling Kit",
        "Smart Array Controller",
        "Entry SAN Cluster Option",
        "RAID Battery Key Cache Kit",
        "CD ROM",
        "CD-RW/DVD-ROM Combo Option Kit",
        "drive"
    ],
    "Server and Networking Equipment": [
        "Rack Mountable Server",
        "Server Blade Enclosure",
        "BladeSystem Entry Bundle",
        "Blade Server Enclosure",
        "Blade",
        "Server",
        "Multi-Server UPS Card",
        "KVM Interface Adapter",
        "KVM Switch",
        "KVM Console Switch",
        "KVM ADAPTERS",
        "KVM ADAPTERS",
        "Server - Tower",
        "Server Adapter",
        "Gigabit Server Adapter",
        "Ethernet Server Adapter",
        "Nic Adapter",
        "Host Bus Adapter (HB)",
        "Network Card",
        "Fibre Channel Cable",
        "Fibre Channel Loop Switch",
        "Host Bus Adapter",
        "Interconnect Switch",
        "SAN Switch",
        "Fibre Channel Switch",
        "Ethernet Adapter",
        "Gigabit Ethernet Network InterfaceAdapter",
        "NIC",
        "Server Adapter Card",
        "Fiber Channel Shortwave Gigabit Interface Converter (GBIC)",
        "Ethernet Card",
        "Fiber Channel Loop Switch",
        "Fiber Channel Shortwave",
        "Module",
        "BladeSystem",
        "Switch",
        "Server Console Cable",
        "Library",
        "Drive Cage Assembly",
        "SCSI CABLE",
        "Expansion Module Rack",
        "transceiver module",
        "I/O Module UPG",
        "Host Adapter",
        "CPU",
        "HUB",
        "Blade server enclosure",
        "Processor Power Module",
        "HB",
        "rack-mountable",
        "Controller",
        "Transceiver",
    ],
    "Power Management and Distribution": [
        "Power Distribution Unit",
        "Redundant Power Supply",
        "AC Power Cord",
        "AC Power Supply",
        "Power Supply Kit",
        "Power Supply Backplane Board",
        "RPS with Fan",
        "CPU Power Module",
        "UPS",
        "PDU",
        "Ext Cache Battery Module",
        "Rail Kit",
        "Power Supply w/Fan",
        "Battery",
        "Power Supply"
    ],
    "Memory and Processing Upgrades": [
        "Memory Module",
        "Memory",
        "SDRAM Memory Kit",
        "Processor Upgrade",
        "Processor Kit",
        "MemoryCartridge",
        "MEM Kit",
        "SDRAM Kit",
        "SDRAM",
        "CPU Kit",
        "Memory Expansion Board",
        "Memory Kit",
        "Processor",
        "Processor Option Kit",
        "SDRAM BBWC Cache Upgrade",
        "SDRAM MEMORY"
    ],
    "Racks, Enclosures, and Mounting Solutions": [
        "Rackmount Option Kit",
        "Rack Server",
        "Rack Rail Kit",
        "Tower to Rack Conversion Kit",
        "Rack Fan Kit",
        "Rackmount Sliding Shelf Kit",
        "Rack-Mount Tape Library AutoLoader",
        "Rack-Mount Library",
        "Rackmount 0 Drive Library",
        "Rack-Mount Tape Library",
        "DRIVE STORAGE ENCLOSURE",
        "ENCLOSURE",
        "Encl",
        "Drive Cage",
        "Baying Kit",
        "Rack Stabilizer Option Kit",
        "Keyboard/Monitor Shelf Kit",
        "Rack Keyboard",
        "LCD Monitor",
        "Rackmount Keyboard",
        "Monitor Utility Shelf",
        "Cable Kit",
        "Rackmount LTO Tape Library AutoLoader",
        "Monitor Utility Shelf Rack Option",
        "Rack Stabilizer Kit",
        "Rack"
    ],
    "Data Transfer and Connectivity": [
        "CD-RW DVD-ROM",
        "CD-RW DVD-ROM Drive",
        "CD / Floppy Assembly",
        "CD / Floppy Assembly Drive Combo",
        "CD-ROM",
        "CD-ROM Drive",
        "DVD-ROM Drive",
        "DVD-ROM drive",
        "CAT5e Cable",
        "Serial Interface Adapter",
        "Patch Panel Connector Cables",
        "Extension Cables",
        "Keyboard/Monitor/Mouse Extension Cables",
        "SCSI Cable",
        "Peripheral Brd"
    ],
    "Media and Accessories": [
        "Tape Array",
        "Tape Library",
        "Tape Drive",
        "DAT",
        "Super DLT",
        "SDLT",
        "SDLT2",
        "SDLT2 Library",
        "LTO Library",
        "SDLT LVD Loader",
        "Tape Drive module",
        "Storage Hub",
        "CD-RW DVD-ROM Combo Option Kit",
        "Motherboard",
        "Sliding Shelf Kit",
        "Keyboard Monitor Shelf"
    ],
    "Cooling and Thermal Management": [
        "Cooling",
        "FAN CHASSIS",
        "Heat Sink",
        "Ventilateur",
        "Cooler",
        "Chassis Fan",
        "Fan",
        "cooling",
        "Cooling Fan"
    ],
    "Miscellaneous and Specialized Equipment": [
        "EMU Board Modular Array",
        "Base",
        "Mezzanine Card",
        "Daughter Board",
        "IDE Crbn Kit",
        "Regulator Board",
        "Voltage Regulator Board",
        "Voltage Regulator Module (VRM)",
        "VRM",
        "CPU Fan",
        "SPS Cable",
        "PC Board (Interface)",
        "Peripheral Board",
        "PS",
        "Switch Box Cable",
        "Interconnect tray",
        "n/a",
        "Console Expansion",
        "Blade System",
        "System Board",
        "BackPlane Board",
        "Controller Board",
        "Controller Adapter Card",
        "Backplane Board",
        "High Airflow Door Insert",
        "Environmental Monitoring Unit (EMU)",
        "Internal Tape Drive",
        "PC Board",
        "Libary Option",
        "SWITCH Kit",
        "Bay Hot Plug",
        "Interconnect",
        "Interconnect Kit",
        "Enclosure",
        "RPS",
        "EMUBOARD MODULAR ARRAY",
        "CTO Blade"
    ]
}

interface_dict = {
    "SCSI Interfaces": [
        "Ultra320 3.5-inch SCSI",
        "Ultra SC",
        "Ultra2/Ultra3 SCSI",
        "Ultra SCSI",
        "(LVD/SE) SCSI",
        "SCSI Ultra 160",
        "(LVD/SE) SCSI",
        "SCSI LVD",
        "SCSI/LVD",
        "Ultra320 LVD SCSI",
        "Ultra3 SCSI",
        "Ultra320 SCSI",
        "U160 SCSI",
        "Ultra SCSI-3",
        "U320 SCSI",
        "SCSI",
        "Ultra2 SCSI",
        "Ultra3",
        "Ultra2",
    ],
    "Fibre Channel Interfaces": [
        "Fibre Channel",
        "PCI to Fibre Channel",
        "Fibre Channel AT (FAT)",
        "FCAL",
        "FC-AL",
        "FC",
        "Fiber Channel"
    ],
    "SAS (Serial Attached SCSI) Interfaces": [
        "Serial Attached SCSI",
        "Serial Attached SCSI (SAS)",
        "Serial AttachedSCSI (SAS)",
        "Serial AT (SAT)",
        "Serial Attached SCSI(SAS)",
        "Serial AttachedSCSI, SAT/SAS",
        "Serial Attached SCSI, Ultra3",
        "PCIe, Serial AttachedSCSI (SAS)",
        "PCI-X SAS",
        "Serial AT(SAT)",
        "SAS",
        "SAT/SAS",
        "SAT",
        "SAS/SATA",
        "SAT-150 / SAS",

    ],
    "Ethernet and Networking Interfaces": [
        "Male DB-9 Connector to RJ-45",
        "Male DB-9Connector to RJ-45",
        "RJ-45",
        "RJ-21 to RJ-45",
        "C-GbE2",
        "C -GBE2",
        "C-GbE",
        "PS/2 and VG",
        "PS/2 and VG",
        "C-GBE"
    ],
    "Other Interfaces": [
        "PCI",
        "PCI-X",
        "PCI- X",
        "LVD/SE",
        "IDE",
        "DIMM 240-pin",
        "LVD",
        "HVD",
        "ATA",
        "IDE, ATA",
        "PCI Express",
        "WU3",
        "SATA",
        "PCIe",
        "AT IDE",
        "LVD SE",
        "Socket AM3+",
        "FAT"
    ]
}


port_dict = {
    2: ["Dual Port", "Dual Port (DP)", "DP", "2-Port", "DUAL-UAT", "2-ports", "2-Ports", "Dual-Port", "Single Port (SP)"],
    16: ["16 Port"],
    8: ["8-Port", "Eight Port"],
    4: ["Quad Port", "4-Port", "2-ports Int/2-ports Ext"],
    1: ["Single Port", "Single Port (SP)"],
    12: ["12-Port"],
    24: ["24 ports"],
    6: ["2-portsInt/4-ports Ext"],
    1: ["SP", "Single Port"]
}


processors_dict = {
    "Intel Xeon Series": [
        "Xeon 800", "Intel Xeon E5-4650L", "Intel Xeon E5503", "Intel Xeon E5-2660", "Intel Xeon",
        "Xeon E5-4650L", "Intel Xeon E5540", "HP Xeon X5272", "Xeon E74830", "Intel Xeon E5430",
        "Intel Xeon E5-2420", "4-Xeon-550", "Intel Xeon E5530", "Xeon 5060", "Intel Xeon process E5345",
        "Xeon E5-2660", "Intel Xeon 5160", "Intel Xeon 5130", "Xeon 5130", "Xeon E5520",
        "Xeon E5335", "Intel Xeon E7-4830", "HP Xeon E5-2650", "Intel Xeon E5-2603", "Intel Xeon DL360",
        "Xeon 5120", "Xeon E5530", "Intel Xeon E7-4850", "Intel Xeon 5060", "Intel Xeon BL20p",
        "Xeon DL360", "Intel Xeon E7220", "Xeon E7-4807", "Intel Xeon E5310", "Xeon E5540",
        "Intel Xeon E5-2650", "Xeon E7220", "Intel Xeon X5272", "Intel Xeon MP DL560", "Xeon DL560",
        "Intel Xeon X5440", "Xeon-mp", "Intel Xeon E5335", "Xeon E5430", "Xeon E5506",
        "Intel Xeon E5405", "Xeon E7440", "Intel Xeon E7-4807", "Xeon 5160", "Xeon E5503",
        "Xeon", "Xeon P3", "Xeon E5-2603"
    ],
    "AMD Opteron/Piledriver Series": [
        "Opteron 6308", "AMD Opteron 8212", "Opteron", "Piledriver FX-4", "Opteron 250",
        "Opteron 8431", "Opteron 8439SE", "AMD Opteron 6308", "AMD Opteron 865", "Opteron 850",
        "AMD Opteron 880", "AMD Opteron 2216HE", "Opteron 880", "Opteron 865"
    ],
    "Pentium Series": [
        "PIII", "Pentium III", "PII", "Pentium II", "Intel Celeron D 352", "Intel Celeron D352",
        "Intel Pentium D 820", "Intel Pentium 4", "P3"
    ]
}

supermarket_aisles_dict = {
    "Coffee and Tea": [
        "coffee", "Tea Brewmaster", "Espresso", "espresso ground coffee",
        "Coffee", "Brewmaster", "Tea For One", "Hot Cups", "Coffee Pods",
        "Footed Iced Tea", "Tea For One set", "Coffeemaker", "Tea Pods", "Coffee and Tea Pods",
        "tea", "teas", "Tea", 'tea for one', "Tea for One", "coffee pod"
    ],
    "Beverages": [
        "sparkling water", "juice", "Immune Defense Drink Mix", "Cold Beverage System",
        "spring water", "sparkling juices", "sparkling waters", "Sparkling Water",
        "water", "Sparkling juices", "Fruit-flavored water", "Juice", "Water", "Spring Water",
        "Sparkling Juices"
    ],
    "Snacks and Breakfast": [
        "fruity snacks", "Microwave Popcorn", "Cookie", "Blow Pops",
        "Blow pops", "candy", "Nuts", "Cocoa Roasted Almonds", "Cookies", "Snack",
        "Breakfast Biscuits", "candies", "snack", "cookies", "oatmeal", "Oatmeal",
        "Trail Mix", "Pistachios", "snacks", "Candy", "Cereal Bars", "fruit snack bars",
        "sandwich cookies", "Cereal Bar", "Candy Honey Bunny Heart Tray", "Granulated Sugar", "Hard Candies",
        "lollipops", "Fruity Snacks", 'Soft & Chewy Mix'
    ],
    "Kitchen and Dining Accessories": [
        "Shredder Lubricant", "Shredder Oil & Lubricant", "Barpac mixer", "Bud Vase",
        "Rocks Glass", "Tea Saucer", "Bread & Butter Plate", "Bud Vases", "BarPac Mixers",
        "Double Old Fashioned Glass", "Cups", "Cup", "cup", "Biscuit Barrel", 'cups'
    ],
    "Sweeteners and Sugars": [
        "Sugar Packets", "sugar", "Pure Sugar Cane", "pure cane sugar", "Natural sugar", "Sugar Portion Packets",
        "No Calorie Sweetener", "Natural Sugar", "Sweetener", "Pure sugar cane"
    ],
    "Health Foods and Protein Powders": [
        "Hemp Protein", "Hemp Protein Powder", "Bean Protein Powder", "Plant Based Protein Powder",
        "Flavored Drink Mix"
    ]
}


home_and_garden_dict = {'Food and Drink Preparation and Cooking Equipment': [
    'Countertop Food Warmers',
    'Slicer',
    'Heavy Duty Range',
    'Convection Oven',
    'Panini/Clamshell Grill',
    'Under Bar Freestanding Drainboard',
    'Countertop Drink Blender',
    'Transportable Gas Grill',
    'Commercial Planetary Mixer',
    'Commercial Microwave',
    'Food Slicers',
    'Countertop Steamer',
    'Underbar Blender Station',
    'Steam Pan',
    'Hot Plate',
    'Convection Steamers',
    'Marmite',
    "Grills",
    'Soup Chafer Marmite',
    'Outdoor Grill',
    'Outdoor Grills',
    'Bakeware Set',
    'Oven Basics',
    'Coffee Brewer',
    'Drop In Griddles',
    'Charbroiler',
    'Range',
    'Boiler',
    'Countertop Gas Fryer',
    'restaurant range',
    'Knife Block Set',
    'Microwaves',
    'Mixer',
    'Hotplate',
    'Gas Griddles',
    'Restaurant Range',
    'Heavy Duty Microwave Oven',
    'Toasters',
    'Food Pan',
    'Professional knives',
    'Charbroilers',
    'Cold Food Bar Work Table',
    'Hot Dog Bun / Roll Warmer',
    'Scale',
    'Fryer',
    'Gas Range',
    'Char-Rock Broiler',
    'Butcher Block Unit',
    'High-Power Blender',
    "Steam Table Pan",
    'Sandwich / Panini Grill',
    'Countertop Griddle',
    'Oven Basics set',
    'Mixers',
    'Electric Griddles',
    'Griddle',
    'Butcher Block',
    'Moist Heat Bun/Food Warmer',
    'Food Slicer',
    'Toaster',
    'Char-Broiler/Oven',
    'Convection Steamer',
    'Radiant Charbroiler',
    'Food Warmers',
    'Toaster Pop-Up'],

    "Ice Machines, Bins and Makers": [
    'Modular Cube Ice Maker',
    'Ice Maker With Bin',
    'Ice Bagger',
    'Cube Ice Machine',
    'Underbar Ice Bin/Cocktail Station',
    'Ice Maker',
    'Ice Makers',
    'Countertop Cube Ice Dispenser',
    'Ice Storage Bin',
    'Cube Ice Maker',
    'Ice Bin with Chute Door',
    'Undercounter Crescent Cube Ice Maker',
    'Ice Maker/Dispenser',
    'Ice Bin/Bottle Section',
    'Ice Chest',
    'Ice Maker/Water Dispenser',
    'Ice Bin',
    'Underbar Ice Bin',
    'Ice Bins',
    'Ice Bin for Machines',
    'Cube Ice Dispenser',
    'Pass-thru Ice Chest',
    'Underbar Ice Bin/Cocktail Unit',
    'Undercounter Nugget Ice Maker',
],


    'Food and Beverage Storage, Dispensers, Serving, Display and Refrigeration': [
    'Raised Liquor Display Unit',
    'Beverage Carrier',
    "freezer refrigerator with Icemaker",
    'Beverage Dispenser',
    "Pass-Thru Back Bar Refrigerator",
    'Backbar Cabinet Refrigerated Beverage',
    'Tea Dispenser',
    'Vending Merchandising Kiosk',
    'Frozen Drink Machine',
    'Refrigerator',
    'Iced Tea/Coffee Dispenser',
    'Refrigerated Beverage Cabinet',
    'Heated Deli Display Case',
    'Case Deli',
    'Heated Deli, Countertop',
    'Serving Counter, Cold Food',
    'Undercounter Refrigerator',
    'Serving Buffet, cold food',
    'Beverage Dispensers',
    'Section Bar Refrigerator',
    'Refrigerator Rack, Roll-In',
    'Open Air Cooler',
    'Refrigerated Backbar Storage Cabinet',
    'Countertop Heated Display Case',
    'Specialty Display Hybrid Merchandiser',
    'Water Cup Dispenser',
    'Display Case',
    'Worktop Freezer',
    'Heat Deli Display Case',
    'Display Case, Heated Deli, Countertop',
    'Display Cases',
    'Roll-In',
    'Refrigerators',
    'Refrigerated Low-Profile Equipment Stand',
    'Freezers',
    'Heated Deli',
    'Refrigerated Back Bar Storage Cabinet',
    'Refrigerator Rack',
    'Back Bar Cabinet, Refrigerated Beverage Cabinet',
    'Coffee Server',
    "Serving Buffet",
    'Cup Dispenser',
    "Serving Trays",
    'Roll-In Refrigerator'],

    'Tableware': ['Bread & Butter Plate',
                  'Bowls',
                  'Coffee Satellite',
                  'Salt / Pepper Shaker',
                  'Wine, Brandy Glass',
                  'Hiball',
                  'Sugarcane Bowls',
                  'Champagne Flutes',
                  'Tumbler',
                  'dinnerware',
                  'Champagne Flute',
                  'Coasters',
                  'Cordial',
                  'Shot Glass',
                  'Tralee stemware',
                  'Salad Plates',
                  'Stemware',
                  'tasting glasses',
                  'Shot Glasses',
                  'Ice Cream Bowl',
                  'Red Wine glass',
                  'Bowl',
                  'Mug',
                  'Teapot',
                  'Mugs',
                  'Salad Plate',
                  'Dinner Plate',
                  'Cutlery',
                  'Goblet',
                  'Goblets',
                  'Tralee Old Fashioned',
                  'European Steak Knife',
                  'Cordial glasses',
                  'Pepper Shaker',
                  'Decanter',
                  'Plates',
                  'Glassware',
                  'Salt Shaker',
                  'Serving Bowl',
                  'Plate',
                  'Wine Glasses',
                  'Platter',
                  'Brandy Glass',
                  'Vases',
                  'Vase',
                  'vase'],

    "Smokers Equipment": [
    'Ashtray',
    'Smoking Receptacle',
    'Trash Can Top Cigarette Receptacle',
    'Urn Cigarette Receptacle',
    'Ash Tray Receptacle',
    'Pole Cigarette Receptacle',
    'Cigarette Receptacle',
    'Container',
    "Smokers' Station",
    'Urn with Weather Shield',
],

    "Dishwashers and related Equipment": [
    'Dishwasher Tabs',
    "dishwasher",
    'Dishwasher Rack',
    'Dishwasher',
    'Dishwashers',
    'dishwasher tabs',
    'Dishwasher Rack, Glass Compartment',
],

    'Cleaning, Sinks, Maintenance Equipment and Trash Disposal': ['Disinfecting Wipes',
                                                                  'tissue',
                                                                  "disinfectant spray",
                                                                  'Sanitary Napkin Receptacle, Plastic Liner Bags',
                                                                  'Drawstring bags',
                                                                  'cleansing scrubber',
                                                                  'Bins',
                                                                  "Jumbo Roll Bath Tissue",
                                                                  'bag',
                                                                  "Sinks",
                                                                  "Undermount Sink",
                                                                  'Hand Cleaner',
                                                                  'Hand/Body Wipes',
                                                                  'Sanitizing Hand Wipes',
                                                                  "Hard Roll Towel",
                                                                  "Sink, undermount",
                                                                  'Bath Tissue',
                                                                  'Replacement Filter',
                                                                  'Sanitizing Wipes',
                                                                  'Lotion Soap',
                                                                  'Drain Board',
                                                                  'Outdoor Decorative Trash Can',
                                                                  'Under Bar Beer Drainer',
                                                                  'Office Air Cleaner',
                                                                  'Office Air Cleaners',
                                                                  'Underbar Trash Station',
                                                                  'Ranger Container',
                                                                  'Pot & Pan Washer',
                                                                  'Sinks with Drainboard',
                                                                  'Underbar Sinks',
                                                                  'Underbar Hand Sink Unit',
                                                                  'Sink',
                                                                  'Underbar Sink',
                                                                  'Pedestal Commercial Hand Sink',
                                                                  'Trash Receptacles',
                                                                  'Sparkle Terrific Trimmers',
                                                                  "Step Can",
                                                                  'Clean Dishtable',
                                                                  'Recycle Bin',
                                                                  'bags',
                                                                  'Plastic Liner Bags',
                                                                  'Cleaning Agents',
                                                                  'Trash Can',
                                                                  'Commercial Trash Can',
                                                                  'Hand/Body Wet Wipe',
                                                                  'glass cleaner',
                                                                  'Plastic Step Trash Can',
                                                                  'Surface Sanitizer',
                                                                  'Fabric Spot Remover',
                                                                  'Laundry Detergent',
                                                                  'kitchen and bath scrubber',
                                                                  'Ash/Trash Receptacle',
                                                                  'Can Liners',
                                                                  'Liquid Drain Cleaner',
                                                                  'Waste Receptacle',
                                                                  'jumbo roll bath tissue',
                                                                  'can liners',
                                                                  'Waste Container w/dolly',
                                                                  'Trash Receptacle',
                                                                  'Floor Shine Cleaner',
                                                                  'Waste Container',
                                                                  'Kitchen Trash Bags',
                                                                  'lotion soap',
                                                                  'Indoor Receptacle',
                                                                  'Plastic Liner Bags for Sanitary Napkin Receptacles',
                                                                  'Waste Bags',
                                                                  'Glass Cleaner',
                                                                  'Oxi-Active Stainlifter',
                                                                  'Paper Towels',
                                                                  'Drain Cleaner',
                                                                  'Disinfectant Spray',
                                                                  "Drawstring Kitchen Bags",
                                                                  'Recycling Bins',
                                                                  'Carpet/Upholstery Spot/Stain Remover',
                                                                  'Kitchen Sink Cleansing Scrubber',
                                                                  'Waste Basket',
                                                                  'Disinfectant/Cleaner',
                                                                  'Hand Sanitizing Wipes',
                                                                  'Hard Roll Paper Towels',
                                                                  'Trash Bags',
                                                                  'Trash Cans',
                                                                  'Degreaser/Cleaner',
                                                                  'Sanitary Napkin Receptacle',
                                                                  'Oven and Grill Cleaner',
                                                                  'Hard Roll Towels',
                                                                  'Hand Sink Unit',
                                                                  'Commercial Hand Sink',
                                                                  'Scullery Sink',
                                                                  'Economy Hand Sink',
                                                                  'Wipes'],

    "Carts, Dollies and Transportation": [
    'Tractor Cart',
    'Food Cart',
    'CamKiosk Cart',
    'Utility Cart',
    'Dish Caddy',
    'Tray & Silver Cart',
    'Bus Cart',
    'Dolly Rack',
    'Utility/Tray Delivery Truck',
    'Transport Carts',
    'Platform Truck',
    'Trash Cart',
    'Cart',
    'Dish Cart',
    'Transport Cabinet',
    'Presentation Carts',
    'Carriers',
    'Presentation Cart',
    'Delivery Cart',
    'Service Cart',
    'Meal Delivery Cart',
    'Cart, Transport Utility',
    'Tray Cart',
    'Dolly',
    'Camdolly with Handle',
    'Camdolly for Camcarriers',
    'Food Carrier Dolly',
    'Cateraide Dolly',
    'Dolly, Dishwasher Rack',
    'Camdolly',
    'Transport Utility',

],

    'Furniture, Storage, Racks and Fixtures': [
    'Rack Compartment',
    'Camshelving Drying Rack',
    'Glass Racks',
    'Camshelving',
    'Camshelving Elements Starter Unit',
    'Camrack Glass Rack',
    'Wire Shelving',
    "Bun Pan Rack Cabinet",
    'All Purpose Racks',
    'Shelving Unit, Wine',
    "Shelf, wire",
    'Camshelving Shelf Plate Kit',
    'Camshelving Mobile Starter Unit',
    "Glass Rack",
    'Tables',
    'Under Bar Glass Storage Unit',
    'Cup Rack',
    'Add-On Shelving Unit',
    'Six Hook Metal Costumer',
    'Glass Rack Storage Unit',
    'Wine Shelving Kit',
    'Underbar Glass Storage Unit',
    'Regal Trays',
    'Regal Tray',
    'Bookcases',
    'Economy Shelf',
    'Shelf',
    'display/storage rack/bookshelf',
    'Shelving Accessories',
    'Equipment Stand',
    'Dishtable',
    'Dining Tables',
    'Base Rack',
    'Shelving',
    'Enclosed Work Table',
    'Work Table',
    'Bookcase',
    'Worktable',
    'Back Bar Cabinet',
    'Countertop',
    'Multipurpose Shelf',
    'Shelving Unit',
    'Camshelving Add-On Unit',
    'Economy Work Table',
    'Food Bars Work Table',
    'Butcher Block Stand',
    'Racks',
    'Caster',
    'Corner Work Table',
    'Automatic Brewers',
    'Camrack PlateSafe',
    'Folding Banquet Table',
    'Dunnage Rack',
    'Medical Containers',
    'seat',
    'Wall Mounted Shelving',
    'Wire Shelf',
    'Beverage Table',
    'bookcase',
    'Camshelving Elements Mobile Starter Unit',
    'Equipment Stand, mobile',
    'Floor Track Shelving Unit',
    'Workstation Desks',
    'Chairs',
    'Work Tables',
    'Flat Countertop',
    'Seat',
    'Add-On Unit',
    'Two Drawer Lateral File',
    'Table Top',
    'Booster Seat',
    'Camshelving Elements Add-On Unit',
    'Shelving Units',
    'Under Bar Storage Unit w/ Sink',
    'Pass-Thru Perlick Station',
    'Folding Table',
    'Desks',
    'Louvered Shelving Unit',
    'Microwave Shelf',
    'Classic Container',
    'Pneumatic Lift Lab Stools',
    'Camshelves',
    'storage for books',
    'Organizer Hutches',
    'organizer hutch',
    'File Cabinets',
    'Six-Hook Metal Costumer',
    'Wine Shelving Kits',
    'Organizer Hutch',
    'Two-Drawer Lateral File',
    'Corner Desk'],

    'Office and Organizational Supplies': [
    'tape',
    'Motivational Print',
    'Time Calculating Recorder',
    'accessories',
    'Workstation Accessories',
    'Poster Board',
    'Graphic Chart Tape'],

    'Accessories, Miscellaneous': [
    'K-Mat Sponge Floor Runner',
    'boards',
    "bathmat",
    "Scraper Mat",
    'Fireplace Screen',
    "fireplace screen",
    "Candelabra",
    'Time recorder',
    'Fireplace Surround Décor',
    'Floor Mat',
    'wire',
    'Chair Mat',
    'Mirror',
    'Mirrors',
    "Post",
    'mirrors',
    'Bath Mat',
    'Stem Caster w/ Brake',
    'Decorative Items',
    'chair mat',
    'Outdoor Scraper Mat']}


office_products_dict = {'Writing/Erasing Instruments and Accessories': ['rollerball pens',
                                                                        'Sharpener',
                                                                        'Rollerball pens',
                                                                        'Gel Ink Rolling Ball Pen',
                                                                        'pens',
                                                                        'eraser',
                                                                        'Eraser',
                                                                        'Correction Tape',
                                                                        'correction tape',
                                                                        'erasers',
                                                                        'Ball Point Pen',
                                                                        'Gel Ink Roller Ball PenBallpoint Pen',
                                                                        'gel ink rolling ball pen',
                                                                        'markers',
                                                                        'Ballpoint Pen',
                                                                        'Roller Ball Stick Waterproof Pen',
                                                                        'Gel Ink Roller Ball Pen',
                                                                        'pencils'],
                        'Paper Products': [
    'pop-up note refills',
    'sticky notes',
    'Sticky notes',
    'multipurpose paper',
    'Ruled paper pads',
    'Paper',
    'design paper',
    'steno notebook',
    'note pads',
    'Medical Security Paper',
    'fiber paper',
    'Index Cards',
    'Note Pads',
    'Steno Book',
    'all-purpose classroom art paper',
    'Composition Book',
    'Legal Ruled Pad',
    'colored paper',
    'Wipe-Off Sentence Strips',
    'book',
    'notebook',
    'Design Suite Paper',
    'Filler Paper',
    'Pop-Up Note Refills',
    'folio/writing pad',
    'Spell-Write Steno Book',
    'ruled pads',
    'Heavyweight paper',
    'Photo Paper',
    'Paper Rolls',
    'steno book',
    'Scratch Pads',
    'Fine Art Paper',
    'graph paper',
    'paper rolls',
    'specialty papers',
    'notebooks',
    'Vellum Title Block/Border',
    'notes cube',
    'Matte Paper',
    'Writing Pads',
    'Paper Roll',
    'paper in a convenient roll',
    'Thermal Facsimile Paper',
    'Notebook',
    'Legal Pads',
    'legal pads',
    'Cross Section Pads',
    'ring-ledger sheets',
    'Business Promotions Sample Pack',
    'computer printout paper',
    'Computer Printout Paper',
    'filler paper',
    'Multipurpose Paper',
    'vellum',
    'Notes Cube',
    'paper',
    'photo paper',
    'Notes',
    'business papers',
    'engineering/architect paper',
    'Construction Paper',
    'vellum with engineer and architect title block/border',
    'Pop Up Note Pad Refills',
    'Felt Sheet Pack',
    'Chart Tablets',
    'Paper Glossy',
    'Reporter/Steno Book',
    'Ruled Pads',
    'Color Card Stock',
    'Colored Cardstock Paper',
    'Computer Paper',
    'sheets',
    'Quadrille Pads',
    'Padfolio',
    'printer paper',
    'Graph Paper',
    'facsimile paper',
    'Sheets for Six-Ring Ledger Binder',
    'original pads',
    'Original Pads',
    'cross-section pads',
    'Art Rolls',
    'Notebook Filler Paper'],
    'Printing, Copying, Laminating': ['Ink cartridges',
                                      'print cartridges',
                                      'Drum Unit',
                                      'Print Cartridge',
                                      'Laser Toner Cartridge',
                                      'Laminating Pouches',
                                      'laminating pouches',
                                      'Toner Cartridge',
                                      'Ribbons',
                                      'toner cartridges',
                                      'printers',
                                      'Toner Print Cartridge',
                                      'Monochrome Printer',
                                      'copiers',
                                      'Solid Ink Color Refills',
                                      'IMAGING UNIT PHASER',
                                      'Drum',
                                      'Thermal Laminator Pouches',
                                      'Wireless Color Inkjet Printer with Scanner, Copier and Fax',
                                      'Photoconductor Kit',
                                      'Printhead',
                                      'ribbons',
                                      'Toner',
                                      'ink',
                                      'inkjet',
                                      'drum',
                                      'Ribbon',
                                      'Ink Cartridges',
                                      'Ethernet Monochrome Printer',
                                      'Cartridge',
                                      'Ink',
                                      'Waste Toner Box',
                                      'Solid Ink Sticks',
                                      'Laser Toner',
                                      'Copier Toner',
                                      'ink tank',
                                      'drum units',
                                      'cartridges',
                                      'laser printer',
                                      'Nylon Ribbons',
                                      'ink cartridges',
                                      'laser toner cartridge',
                                      'toner cartridge',
                                      'Ink Glossy Photo Paper Combo Pack',
                                      'TONER CARTRIDGE',
                                      'Mono Laser All-in-One Printer',
                                      'fax machines',
                                      'laminators',
                                      'toners',
                                      'photoconductor unit',
                                      'Laminating Folders',
                                      'Thermal ink ribbons',
                                      'Ink Cartridge',
                                      'Ink Tank',
                                      'cartridge',
                                      'Solid Ink',
                                      'Print CartridgeToner Cartridge',
                                      'Nylon Ribbon',
                                      'Nylon & 4-Color Ribbons',
                                      'Thermal Ink Ribbon Cartridge',
                                      'Print labels',
                                      'Fuser Unit',
                                      'laser drum',
                                      'Toner/Developer/Drum Cartridge',
                                      'Thermal Laminating Pouches',
                                      'MICR Toner',
                                      'Printer Labels'],
    'Folders, Dividers, Indexing, Flags, Labels, Cases, Holders, Bindings': [
    'Expanding File',
    'Sheet Protector',
    'Business Card Binder',
    'Sheet Protectors',
    'Bulk Index System',
    'Removable Multi-Use Labels',
    'Two-Pocket Portfolio',
    'Page Flags in Dispenser',
    'removable labels',
    'removable multi-use labels',
    'File Folder Labels',
    'Page Markers',
    'tape flags',
    'Tape Flags',
    'Marking Flags',
    'Paper Fasteners',
    'heavy-duty binder',
    'Heavy-Duty Binder',
    'Heavy-duty binders',
    'Card Guides',
    'Preprinted Dividers',
    'paper dividers',
    'Index Tabs',
    'index tabs',
    'Tab Dividers',
    'Print-On Dividers',
    'Folio/Writing Pad',
    'report covers',
    'folders',
    'Classification Folder',
    'Business Card Holder',
    'Interior File Folders',
    'interior folders',
    'Paper Index Dividers',
    'dividers',
    'clear sheet protector',
    'Clear Sheet Protector',
    'tabs',
    'File Folders',
    'business card binder',
    'file sorter',
    'File sorter',
    'shop ticket holders',
    'interior file folders',
    'color flags',
    'Color Flags',
    'No-Punch Report Cover',
    'Index System',
    'Card Holder',
    'folder',
    'Index Card File Holds',
    'Folder',
    'Index Dividers',
    'preprinted dividers',
    'No-Punch Report Covers',
    'desktop folder',
    'Desktop Folder',
    'Sheet Dividers',
    'Project Folders',
    'Report Covers',
    'Poly Pockets with Index Tabs',
    'Versatile case',
    'Rolling Notebook Case',
    'file folder labels',
    'Wire Bindings',
    'index file tabs',
    'Index File Tabs',
    'Dividers',
    'File Tabs',
    'tab indexes',
    'Tab Indexes',
    'Shop Ticket Holders',
    'Classification Folders',
    'file tabs'],
    'Office Furniture and Ergonomic Products': ['wrist rests',
                                                'Filing System',
                                                'Doodle Desk Pad',
                                                'files',
                                                'Gel Wrist Rest and Mouse Pad',
                                                'Desk File',
                                                'Pencil Drawer Accessory',
                                                'file cabinets',
                                                '8-Drawer',
                                                'File Cabinet',
                                                'File Shelf',
                                                'Pedestal',
                                                'Deskside File Cart',
                                                'mouse pads',
                                                'mouse',
                                                'Lateral File Cabinet',
                                                'sign holder',
                                                'Deskside Recycling Container',
                                                'stacking sorter',
                                                'Mobile File Cabinet',
                                                'Filing system',
                                                'Suggestion Box',
                                                'mobile machine stand',
                                                'keyboard trays',
                                                'Laptop Stand/Holder',
                                                'Base/Media Cart',
                                                'Wall File Pocket',
                                                'Letter Tray',
                                                'suggestion box',
                                                'Mailroom sorter',
                                                'file cabinet',
                                                'File Cart',
                                                'keyboard wrist rest',
                                                'mobile stands',
                                                'Storage/Transfer File',
                                                'Open Shelf Files',
                                                'Stacking Sorter',
                                                'Lateral Files',
                                                'Mobile Machine Stand',
                                                'gel wrist rest and mouse pad',
                                                'keyboard tray',
                                                'filing system',
                                                'Sign Holder',
                                                'Keyboard Wrist Pillow',
                                                'Deskside Letter/Legal Mobile File Cart',
                                                'Sorter',
                                                'laptop stand',
                                                'letter trays',
                                                'Mobile File Cart',
                                                'shelving solutions',
                                                'Lectern/Media Cart',
                                                'Storage Files',
                                                'Pedestal Desk',
                                                'Keyboard Tray',
                                                'Deskside Recycling Containers',
                                                'drawer units'],
    'Board and Presentation Materials': ['Whiteboard',
                                         'Laser Pointer',
                                         'easel',
                                         'Dry-Erase Board',
                                         'dry erase marker board',
                                         'easel pads',
                                         'Bulletin Board',
                                         'easel pad',
                                         'Board',
                                         'Display Boards',
                                         'Magnetic Presentation Easel',
                                         'display boards',
                                         'whiteboards',
                                         'easel boards',
                                         'Dry Erase Board',
                                         'projectors',
                                         'display board',
                                         'Presentation Board',
                                         'Easel Pads',
                                         'Dry Erase Marker Board',
                                         'Dry Erase Easel',
                                         'whiteboard',
                                         'bulletin board',
                                         'Display Board'],
    'Shipping and Packaging, Envelopes': ['Window Envelope',
                                          'mailer',
                                          'window envelope',
                                          'catalog envelope',
                                          'Catalog Envelope',
                                          'mailers',
                                          'Strung tags',
                                          'Bubble Mailers',
                                          'Postage Meter Ink',
                                          'Carton Sealing Tape',
                                          'Envelopes',
                                          'Cushioned Mailers',
                                          'Paper Bag',
                                          'Filament Tape',
                                          'labels',
                                          'Strung Shipping Tags',
                                          'envelopes',
                                          'Catalog Envelopes',
                                          'packing tapes',
                                          'Envelope',
                                          'Shipping Cartons',
                                          'envelope',
                                          'strung tags',
                                          'shipping tags',
                                          'Clasp Envelope',
                                          'Paper Bags & Sacks',
                                          'Expansion Envelopes',
                                          'Address Labels',
                                          'postal scales',
                                          'book pockets',
                                          'Library Book Card Pockets',
                                          'Mailer',
                                          'Shipping Tags',
                                          'Tape Refill Rolls'],
    'Cleaning and Maintenance': ['Whiteboard Eraser',
                                 'Chalkboard/Dry Erase Eraser',
                                 'cleaning wipes',
                                 "Scrub 'n Strip Pad",
                                 'Can Liners',
                                 'Premoistened wipes',
                                 'Waste bags',
                                 'waste bags',
                                 'Plastic bags',
                                 'Extra-Heavy Bags',
                                 'Shredder Bags',
                                 'Cleaning Wet Wipes',
                                 'scouring products',
                                 'Push Dispenser'],
    'Stationery and Desk Accessories': ['Tape Dispenser',
                                        'Crown Staples',
                                        'Rubber Bands',
                                        'Wire Clips',
                                        'staples',
                                        'hole punches',
                                        'Hole reinforcements',
                                        'Seven-Hole Punch',
                                        'Tape',
                                        'Two-to-Three-Hole Adjustable Punch',
                                        'Tape Gun',
                                        'General Purpose Masking Tape',
                                        'Desktop hole punch',
                                        'Clipboard',
                                        'Stapler',
                                        'Paper Clip',
                                        'tape',
                                        'two-to-three-hole adjustable punch',
                                        'stamp dispenser',
                                        'Stamp Dispenser',
                                        'Message Stamp',
                                        'dispenser',
                                        'tape gun',
                                        'Paper Trimmer',
                                        'Laser Trimmer',
                                        'laser',
                                        'desktop hole punch',
                                        'Mouse',
                                        'stapler',
                                        'Office Tape',
                                        'Masking Tape',
                                        'staplers',
                                        'Staplers',
                                        'Tape dispensing',
                                        'stamping',
                                        'Electric Desktop Sharpener',
                                        'Recycled Clipboards',
                                        'hole-punch',
                                        'Hole Reinforcements'],
    'Stickers and Badges': ['Name Badges',
                            'name badges',
                            'stickers',
                            'ID Card Reels',
                            'Reward Stickers',
                            'Applause STICKERS',
                            'reward stickers'],
    'Time, Planning and Organization': ['Weekly Appointment Book',
                                        'wall calendar',
                                        'desk calendar refill',
                                        'Desk Calendar Refill',
                                        'Wall Calendar',
                                        'Time Recorder Weekly Cards',
                                        'Class Record Book',
                                        'Weekly Time Cards',
                                        'desk calendars',
                                        'Erasable Wall Planner',
                                        'monthly calendars'],
    'Miscellaneous': ['Paper Plates',
                      'Notebook Thesaurus',
                      'document frame',
                      'Naproxin Tablets',
                      'LED OPEN Sign',
                      'Cash Box',
                      'Open Rotary Card File',
                      'power pad',
                      "Power Pad",
                      'Time Clock Cards',
                      'Visitor Register Book',
                      'Bill of Lading',
                      'Receipt',
                      'Employment Application',
                      'Purchase Order Book',
                      'Phone Call Book',
                      'Forms Application',
                      'Ear Plugs',
                      'noise reduction',
                      'Earplugs',
                      'Document Frame',
                      'Executive Plaque',
                      'plates',
                      'LED',
                      'Hazardous Material Short Form',
                      'Packing Slip Book',
                      'Money Receipt Book']}


color_mapping = {
    "Black": ["Black, Sapphire",
              "Black/Chrome",
              "Black/Black",
              "Black, Chrome",
              "black marble",
              "Black/White",
              "Black and Purple",
              "black/red",
              "Black/Silver",
              "Matte Black",
              "Charcoal",
              "Black/Gray",
              "Black",
              "black",
              "Onyx"],

    "White": [
        "Bright White",
        "Brightly colored",
        "WE",
        "Stardust White",
        "Bright white",
        "white",
        "White",
        "white or clear",
        "White/Canary"],


    "Brown": ["Medium Cherry",
              "Gold Brown",
              "Putty",
              "Brown Kraft",
              "Light Cherry",
              "Walnut",
              "Golden Brown",
              "Mahogany",
              "Natural Cherry",
              "Sierra Cherry",
              "coffee beige",
              "Brown",
              "brown",
              "walnut",
              "Light Brown",
              "dark brown",
              "Cherry",
              "Light Cherry/Black",
              "Natural Cherry/Slate Gray",
              "Med Cherry",
              "Rosewood/Black"],

    "Green": ["Bright Green",
              "Sherwood Green",
              "empire green with desert brown",
              "Empire Green",
              "green",
              "Sherwood green",
              "Neon Lime Green",
              "empire green"],

    "Beige": ["Putty",
              "BEIG",
              "beige",
              "Beige",
              "Pastel Ivory",
              "Manila",
              'Cameo Buff'],

    "Gray and Metallic": ["Gray/Green",
                          "metallic",
                          "Smoke",
                          "Onyx",
                          "Dark Gray",
                          "Light Gray",
                          "Graphite",
                          "Metallic Blue",
                          "Light Gray/Light Gray",
                          "Silver",
                          "Metallic Silver",
                          "light gray",
                          "Granite Gray",
                          "brushed graphite",
                          "gray",
                          "GRAY",
                          "soft gray",
                          "speckled gray",
                          "gray with dove gray",
                          "granite gray",
                          "Slate Gray",
                          "gray with Silverstone gray",
                          "Gray",
                          "metallic grey",
                          "architectural bronze with glacier gray",
                          "Silverstone gray",
                          "silver",
                          "metallic grey",
                          "blending warm metallics",
                          "copper",
                          "lustrous gold",
                          "satin brass",
                          "Bronze",
                          "hammertone bronze",
                          "bronze",
                          "chromate",
                          "gold"],

    "Blue": ['Black, Sapphire',
             "Ice Blue",
             "Blue/Black",
             "Blue Black",
             "Navy/Black",
             "Bright Blue",
             "Lavender",
             "Blue, White",
             "Sky Blue",
             "Non-repro blue",
             "Blue/Pink",
             "Metallic Blue",
             "Cyan",
             "Navy Blue",
             "Blue",
             "blue",
             "navy blue",
             "slate blue"],

    "Red": ["Ruby Red",
            "Magenta",
            "Red & Blue",
            "red and blue",
            "cranberry",
            "Cranberry",
            "hot red",
            "brick red",
            "red",
            "Mahogany",
            "Red Opaque"],

    "Pink": ["Magenta",
             "Carnation Pink",
             "Roses Pink",
             "Rose Confetti",
             "Cheeky Pink",
             "pink and yellow"],

    "Yellow": ["yellow",
               "Canary",
               "Canary Yellow"],

    "Purple": ["Purple",
               "Violet"],
    "Assorted/Others": ["Mono",
                        "Monochrome",
                        'burgundy, pink and yellow',
                        'silver, red, yellow and blue',
                        "Black, Cyan, Magenta, Yellow",
                        "red, blue, clear and purple",
                        "Translucent Assorted",
                        "blue, green, yellow & red",
                        "Assorted Translucent",
                        "Vibrant colored",
                        "multicolored",
                        "Neon Green, Neon Orange, Neon Pink, Neon Citrus",
                        "Opaque",
                        "spring green, carnation, sky blue, canary yellow",
                        "multiple",
                        "black, blue, red, green and clear",
                        "Monochrome",
                        "Brightly colored",
                        "Assorted Fluorescent",
                        "lively pastels",
                        "Blue/Pink",
                        "Assorted",
                        "Pastel-colored",
                        "Assorted Bright",
                        "Vibrant",
                        "Mono",
                        "Pastel",
                        "bold and vibrant colors",
                        "vibrant colors",
                        "Multicolored",
                        "Five Neon Colors",
                        "Ribbon Candy",
                        "fluorescent pink, fluorescent green and fluorescent orange",
                        "green, orange, pink, citrus and watermelon"],

    "No Color/Transparent": ["Bright translucent",
                             "invisible",
                             "Crystal clear",
                             "Translucent Assorted",
                             "Assorted Translucent",
                             "Clear",
                             'Natural',
                             "Transparent",
                             "white or clear",
                             "Mono",
                             "clear",
                             "see-through",
                             "Translucent"]
}


def recategorize_product_type(product_type, equipment_dict):
    for category, values in equipment_dict.items():
        if product_type in values:
            return category
    return product_type


def recategorize_product_type_multiple(product_type, color_mapping):
    categories = []
    for category, values in color_mapping.items():
        if product_type in values:
            categories.append(category)
    return categories if categories else [product_type]


def convert_rotational_speed_to_numeric(value):
    # Handle 'n/a' or similar non-numeric values
    if value.lower() == 'n/a':
        return value  # Keeps 'n/a' as is

    # Convert K values to thousands and remove 'RPM'
    value = re.sub(r'(\d+(\.\d+)?)(\s?K)',
                   lambda x: str(int(float(x.group(1)) * 1000)), value, flags=re.IGNORECASE)
    value = re.sub(r'\s?RPM', '', value, flags=re.IGNORECASE)

    return value


def convert_to_grams(weight_str):
    """
    Convert weight to grams. Supports ounces (oz), pounds (lbs, Pound), and kilograms (kg).
    If the unit is not one of these, the original value is returned.
    Fractions and various formats are also handled.
    """

    # Conversion factors
    oz_to_g = 28.3495
    lb_to_g = 453.592
    kg_to_g = 1000

    # handle 2 special cases
    special_cases = {
        "8-1 3/4 oz": "397",
        "16-1 1/2 oz": "680",
        "14-3/4-oz.": "419",
        "4/5 oz": "23"
    }
    if weight_str in special_cases:
        return special_cases[weight_str]

    else:
        # Updated regular expression to handle more formats
        weight_str = weight_str.replace("-", " ")
        weight_str = weight_str.replace(';', '')

        regex = r'(?:(\d+(?:\.\d+)?)\s*)?(?:(\d+)\/(\d+))?\s*(oz|ounce|lbs|lb|pound|pounds|kg|g|in|ounces)?'

        matches = re.match(regex, weight_str)

        if not matches:
            return weight_str  # Return original string if no match

        whole_number, numerator, denominator, unit = matches.groups()

        # Calculate the weight in grams
        if whole_number is not None:
            weight = float(whole_number)
        else:
            weight = 0

        if numerator is not None and denominator is not None:
            weight += float(numerator) / float(denominator)

        if unit in ['oz', 'ounces', 'ounce']:
            return round(weight * oz_to_g)
        elif unit in ['lbs', 'lb', 'pound', 'pounds']:
            return round(weight * lb_to_g)
        elif unit in ['kg']:
            return round(weight * kg_to_g)
        elif unit == "g":
            return weight_str.replace("g", "").strip()
        else:
            return weight_str


def convert_to_kg(weight_str):
    """
    Convert weight to kilograms. pounds (lbs, Pound), mil.
    If the unit is not one of these, the original value is returned.
    Fractions and various formats are also handled.
    """

    # Conversion factors
    mil_to_kg = 28.3495
    lb_to_kg = 0.453
    g_to_kg = 0.001

    # handle 2 special cases
    special_cases = {
        "16-24": "10.8",
        "Lbs. 16-24": "10.8",
    }
    if weight_str in special_cases:
        return special_cases[weight_str]

    # Updated regular expression to handle more formats
    weight_str = weight_str.replace("-", " ")

    regex = r'(?:(\d+)\s*)?(?:(\d+)\/(\d+))?\s*(oz|ounce|lbs|lb|pound|pounds|kg|g|in|ounces)?'

    matches = re.match(regex, weight_str.lower().replace(
        ';', '').replace('.', ''))

    if not matches:
        return weight_str  # Return original string if no match

    whole_number, numerator, denominator, unit = matches.groups()
    
    # Calculate the weight in grams
    if whole_number is not None:
        weight = float(whole_number)
    else:
        weight = 0

    if numerator is not None and denominator is not None:
        weight += float(numerator) / float(denominator)

    elif unit in ['lbs', 'lb']:
        return round(weight * lb_to_kg, 1)
    elif unit in ['g']:
        return round(weight * g_to_kg, 1)

    else:
        return weight_str


def convert_to_cm(value_str):
    # Function to convert a fraction to decimal
    def fraction_to_decimal(frac_str):
        # Handle a whole number followed by a fraction
        if '-' in frac_str and not frac_str.endswith("-"):
            whole, fraction = frac_str.split('-')
            return float(whole) + float(fractions.Fraction(fraction))
        elif frac_str.endswith("-"):
            # Remove trailing hyphen if it exists
            return float(frac_str.rstrip("-"))
        else:
            try:
                # Attempt to convert direct decimal or whole number
                return float(frac_str)
            except ValueError:
                # Handle fractions
                return float(sum(fractions.Fraction(s) for s in frac_str.split()))

    value_str = '0' + value_str if value_str.startswith('.') else value_str

    if value_str == '23 - 1/2"':
        return "59.69"

    yards_match = re.match(
        r"(\d+(\.\d+)?)\s*(yd|yards|Yards|yds?)\.?", value_str)
    if yards_match:
        yards_val = float(yards_match.group(1)) * 91.44  # Convert yards to cm
        return f"{round(yards_val, 1)}"

    # Check for feet (ft or ft.) and convert to inches
    feet_match = re.match(r"(\d+)\s*ft\.?", value_str)
    if feet_match:
        feet_val = int(feet_match.group(1)) * 12  # Convert feet to inches
        return f"{round(feet_val * 2.54, 1)}"

    # Check for feet and inches (' and ")
    feet_and_inches = re.match(r"(\d+)'(?:\s+(\d+)(?:\"|\s|$))?", value_str)
    if feet_and_inches:
        feet = int(feet_and_inches.group(1))
        inches = int(feet_and_inches.group(
            2)) if feet_and_inches.group(2) else 0
        total_inches = feet * 12 + inches
        return f"{round(total_inches * 2.54, 1)}"

    decimal_match = re.match(r"(\d+\.\d+)\s*?(\")?(cm|in)?", value_str)
    if decimal_match:
        decimal_val = float(decimal_match.group(1))
        return f"{round(decimal_val * 2.54, 1)}"

    mm_match = re.match(r"(\d+(\.\d+)?)\s*mm", value_str, re.IGNORECASE)
    if mm_match:
        mm_val = float(mm_match.group(1)) * 0.1  # Convert mm to cm
        return f"{round(mm_val, 1)}"

    # Check for meters (m) and convert to cm
    m_match = re.match(r"(\d+(\.\d+)?)\s*m", value_str, re.IGNORECASE)
    if m_match:
        m_val = float(m_match.group(1)) * 100  # Convert m to cm
        return f"{round(m_val, 1)}"

    # Identify if the input is a range
    if ' to ' in value_str:
        value_str = value_str.replace(' to ', ' - ')

    if ' - ' in value_str:
        start_str, end_str = value_str.split(' - ')
        start_match = re.match(r"([\d\s\-/]+)\s*?(\")?(cm|in)?", start_str)
        end_match = re.match(r"([\d\s\-/]+)\s*?(\")?(cm|in)?", end_str)
        if start_match and end_match:
            start_val = fraction_to_decimal(start_match.group(1))
            end_val = fraction_to_decimal(end_match.group(1))
            unit = start_match.group(3) or end_match.group(3)
            if not unit or unit == 'in':
                return f"{round(end_val * 2.54, 1)}"
            else:
                return f"{round(end_val, 1)}"

    # Handle single measurement
    match = re.match(r"([\d\s\-/]+)\s*?(\")?(cm|in)?", value_str)
    if match:
        number_str, _, unit = match.groups()
        value = fraction_to_decimal(number_str)
        if not unit or unit == 'in':
            return f"{round(value * 2.54, 1)}"
        else:
            return f"{round(value, 1)}"

    return value_str


def retail_upc_to_manufacturer_code(upc_str):
    return upc_str[1:6]


def extract_digit_from_pack_quantity(quantity_str):
    """
    Extracts numeric values from the given string and returns them as a list of integers.
    Removes commas from numbers and returns all found numeric values.
    If no numbers are found, returns the original string.
    """
    special_cases = {
        "Five dividers per pack": "5",
        "trio": "3",
        "Ten envelopes per pack": "10",
        "Two-Pen Pack": "2",
        "One set": "1",
        "Dozen Pen": "12",
        "four matching envelopes": "4",
        "Two Pack": "2"
    }

    if quantity_str in special_cases:
        return [(special_cases[quantity_str])]
    else:
        # Find all matches for numeric patterns, including those within text
        numbers = re.findall(r'\d[\d,]*', quantity_str)
        if numbers:
            # Convert to integers while removing commas
            return [int(n.replace(',', '')) for n in numbers if n.replace(',', '').isdigit()]
        else:
            # Return the original string if no numbers are found
            return [quantity_str]


product_categories_jewelry = {
    "Rings/Bands": [
        "Wedding Ring", "wedding band", "Wedding bands", "Wedding Band",
        "Wedding Bands", "Wedding band", "Wedding ring", "wedding ring",
        "wedding bands", "Wedding rings", "Halo engagement ring",
        "Sidestone engagement ring", "Three stone engagement ring",
        "Engagement Ring", "engagement rings", "Engagement rings",
        "Halo ring", "Halo Diamond Engagement Ring", "engagement ring",
        "Sidestone", "Halo", "Fashion Rings", "Fashion rings", "fashion rings", "Halo, Sidestone engagement ring",
        "fashion ring", "ring",  "Sidestone, Halo engagement ring"],
    "Bracelets": [
        "Bangle", "Gemstone bracelet", "Cuff", "bracelets", "bracelet",
        "Bangle bracelet", "Bracelets", 'Bangle, Gemstone bracelet', 'Cuff, Gemstone bracelet'
    ],
    "Necklace/Chain": [
        "Necklace", "Chain", "Gemstone necklace", "necklaces", "chain", "Necklaces", "Chain, Gemstone necklace"
    ],
    "Earrings": [
        "earrings", "Earrings"
    ],
    "Miscellaneous": [
        "Vase", "Perfume Bottle", "Heart Tray", "Picture Frame", "Frame",
        "Bud Vase", "Candlesticks", "Photo Frame", "Platter", "Heart Box",
        "pilsners", "Hiball", "Champagne Toasting Flutes", "Brandy", "Flute",
        "bar essentials", "Pilsner", "flute", "Wedding Vows Flute", "Pilsners", "watch", "Candlestick"
    ]
}


def replace_short_forms(text):
    # Replace generations
    for i in range(1, 10):
        text = text.replace(f"Gen{i}", f"Generation {i}")
        text = text.replace(f"G{i}", f"Generation {i}")

    # Replace DDR versions
    text = text.replace("DDR4", "Double Data Rate 4")
    text = text.replace("DDR2", "Double Data Rate 2")
    text = text.replace("DDR", "Double Data Rate")

    # Replace AIT versions
    text = text.replace("AIT-3", "Advanced Intelligent Tape 3")
    text = text.replace("AIT-2", "Advanced Intelligent Tape 2")
    text = text.replace("AIT-1", "Advanced Intelligent Tape 1")
    text = text.replace("AIT2", "Advanced Intelligent Tape 2")

    # Replace DDS4
    text = text.replace("DDS4", "Digital Data Storage 4")

    # Replace LTO2
    text = text.replace(
        "LTO2", "Linear Tape-Open 2").replace("LTO 2", "Linear Tape-Open 2")

    # Replace DLT1
    text = text.replace("DLT1", "Digital Linear Tape 1").replace(
        "DLT 1", "Digital Linear Tape 1")

    # Replace PII (Pentium II)
    text = text.replace("PII", "Pentium II")

    # Replace Piledriver FX-4
    text = text.replace("Piledriver FX-4",
                        "Piledriver FX-4 Core Processor")

    return text


def get_long_brand_name(text):
    if text == "HP Pro":
        return "Hewlett-Packard ProLiant"
    else:
        text = text.replace("HPE", "Hewlett-Packard Enterprise")
        text = text.replace("HP", "Hewlett-Packard")
        text = text.replace("IBM", "International Business Machines")
        text = text.replace("EMC", "Egan, Marino Corporation")
        text = text.replace("AMD", "Advanced Micro Devices")
        return text


def normalize_core_number(value):
    core_mapping = {
        "QC": 4,
        "4-core": 4,
        "10-core": 10,
        "2-core": 2,
        "Single-core": 1,
        "6-core": 6,
        "8-core": 8,
        "DC": 2,
        "Quad Core": 4,
        "Quad-Core": 4,
        "Dual Core": 2,
        "4-Core": 4
    }

    return core_mapping.get(value)


def delete_marks(id):
    # Remove all non-alphanumeric characters
    return re.sub(r'[^a-zA-Z0-9]', '', id)


def normalize_data(split="test", normalization_params=['Name Expansion', 'Numeric Standardization', 'To Uppercase', 'Substring Extraction', 'Product Type Generalisation', 'Unit Conversion', 'Color Generalization', 'Name Generalisation', 'Unit Expansion', 'To Uppercase', 'Delete Marks']):

    data = []

    if split == "test":
        jsonl_path_test = f'data/processed_datasets/old_structure/wdc/test.jsonl'
        with open(jsonl_path_test, "r", encoding="utf-8") as file:
            for line in file:
                json_obj = json.loads(line)
                data.append(json_obj)

    elif split == "train_1.0":
        jsonl_path_train = f'data/processed_datasets/old_structure/wdc/train.jsonl'
        with open(jsonl_path_train, "r", encoding="utf-8") as file:
            for line in file:
                json_obj = json.loads(line)
                data.append(json_obj)

    elif split == "train_0.2":
        jsonl_path_train = f'data/processed_datasets/old_structure/wdc/train_0.2.jsonl'
        with open(jsonl_path_train, "r", encoding="utf-8") as file:
            for line in file:
                json_obj = json.loads(line)
                data.append(json_obj)

    keys_missing = []
    normalized_attributes = {}
    for product in data:
        category = product['category']
        if category not in normalized_attributes:
            normalized_attributes[category] = []
    for product in data:
        category = product['category']
        if category == "Computers And Accessories":
            for attribute, values in product['target_scores'].items():
                if attribute == "Capacity" and "Unit Expansion" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            # Transforming capacity units
                            if value_key == "2 X 2GB":
                                new_key = "4 Gigabytes"
                            elif value_key == "2x2GB":
                                new_key = "4 Gigabytes"
                            elif value_key == "2GB (1x2GB)":
                                new_key = "2 Gigabytes"
                            elif value_key == "2 x 512 MB":
                                new_value = "1024 Megabytes"
                            elif value_key == "1x4GB":
                                new_key = "4 Gigabytes"
                            else:
                                new_key = value_key.replace("-", " ").replace('GB', ' Gigabytes').replace('Gb', ' Gigabytes').replace(
                                    'TB', ' Terabytes').replace('MB', ' Megabytes').replace('mb', ' Megabytes').replace('W', ' Watts')
                                new_key = re.sub(
                                    r'(\d+)\s?G\b', r'\1 Gigabytes', new_key)

                                # Ensuring a space between the numeric value and the unit
                                new_key = re.sub(
                                    r'(\d)([A-Za-z])', r'\1 \2', new_key)
                                new_key = re.sub(' +', ' ', new_key)

                                if new_key not in new_values:
                                    new_values[new_key] = {
                                        'pid': [], 'score': 1}

                                new_values[new_key]['pid'].extend(
                                    value_details['pid'])

                                if new_key == value_key:
                                    keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    # Update the Capacity values with the recategorized keys
                    product['target_scores']['Capacity'] = new_values
                    normalized_attributes[category].append('Capacity')

                if attribute == "Generation" and "Name Expansion" in normalization_params:
                    new_values = {}
                    for key, value_details in values.items():
                        if key != "n/a":
                            new_key = replace_short_forms(key)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}
                            new_values[new_key]['pid'].extend(
                                value_details['pid'])
                            if new_key == key:
                                keys_missing.append(new_key)
                        else:
                            new_values[key] = value_details
                    product['target_scores']['Generation'] = new_values
                    normalized_attributes[category].append('Generation')

                if attribute == "Part Number" and "Delete Marks" in normalization_params:
                    new_values = {}
                    for key, value_details in values.items():
                        if key != "n/a":
                            new_key = delete_marks(key)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}
                            new_values[new_key]['pid'].extend(
                                value_details['pid'])
                            if new_key == key:
                                keys_missing.append(new_key)
                        else:
                            new_values[key] = value_details
                    product['target_scores']['Part Number'] = new_values
                    normalized_attributes[category].append('Part Number')

                if attribute == "Product Type" and "Product Type Generalisation" in normalization_params:
                    new_values = {}
                    for key, value_details in values.items():
                        if key != "n/a":
                            new_key = recategorize_product_type(
                                key, equipment_dict_computers)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}
                            new_values[new_key]['pid'].extend(
                                value_details['pid'])
                            if new_key == key:
                                keys_missing.append(new_key)
                        else:
                            new_values[key] = value_details
                    product['target_scores']['Product Type'] = new_values
                    normalized_attributes[category].append('Product Type')

                if attribute == "Manufacturer" and "Name Expansion" in normalization_params:
                    new_values = {}
                    for key, value_details in values.items():
                        if key != "n/a":
                            new_key = get_long_brand_name(key)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}
                            new_values[new_key]['pid'].extend(
                                value_details['pid'])
                            if new_key == key:
                                keys_missing.append(new_key)
                        else:
                            new_values[key] = value_details
                    product['target_scores']['Manufacturer'] = new_values
                    normalized_attributes[category].append("Manufacturer")

                # Handling 'Interface' attribute
                if attribute == "Interface" and "Name Generalisation" in normalization_params:
                    new_values = {}
                    for key, value_details in values.items():
                        # Check if the value is not 'n/a'
                        if key != "n/a":
                            new_key = recategorize_product_type(
                                key, interface_dict)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}
                            new_values[new_key]['pid'].extend(
                                value_details['pid'])
                            if new_key == key:
                                keys_missing.append(new_key)
                        else:
                            new_values[key] = value_details

                    product['target_scores']['Interface'] = new_values
                    normalized_attributes[category].append("Interface")

                if attribute == "Cache" and "Unit Expansion" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            if value_key == "2MB-2X1MB":
                                new_key = "2 Megabytes"
                            elif value_key == "256":
                                new_key = "256 Megabytes"
                            elif value_key == "512":
                                new_key = "512 Megabytes"
                            elif value_key == "8MB-2x4MB":
                                new_key = "8 Megabytes"
                            elif value_key == "4MB L2 Cache, 8MB L3 Cache":
                                new_key = "12 Megabytes"
                            elif value_key == "2MB(2x1MB)":
                                new_key = "2 Megabytes"
                            elif value_key == "2MB(2x1MB)":
                                new_key = "2 Megabytes"
                            elif value_key == "2M":
                                new_key = "2 Megabytes"
                            elif value_key == "1M":
                                new_key = "1 Megabytes"

                            else:
                                new_key = value_key.replace("-", " ").replace('GB', ' Gigabytes').replace(
                                    'KB', ' Kilobytes').replace('MB', ' Megabytes').replace('mb', ' Megabytes')
                                new_key = re.sub(
                                    r'(\d+)\s?G\b', r'\1 Gigabytes', new_key)
                                new_key = re.sub(
                                    r'(\d+)\s?K\b', r'\1 Kilobytes', new_key)
                                new_key = re.sub(
                                    r'(\d)([A-Za-z])', r'\1 \2', new_key)
                                new_key = re.sub(' +', ' ', new_key)

                            if new_key == value_key:
                                keys_missing.append(new_key)

                            if new_key not in new_values:
                                new_values[new_key] = {
                                    'pid': [], 'score': value_details['score']}
                            new_values[new_key]['pid'].extend(
                                value_details['pid'])
                        else:
                            new_values[value_key] = value_details

                    product['target_scores']['Cache'] = new_values
                    normalized_attributes[category].append("Cache")

                # Handling 'Ports' attribute
                if attribute == "Ports" and "Numeric Standardization" in normalization_params:
                    new_values = {}
                    for key, value_details in values.items():
                        if key != "n/a":
                            new_key = recategorize_product_type(key, port_dict)
                            if new_key not in new_values:
                                new_values[new_key] = {
                                    'pid': [], 'score': value_details['score']}
                            new_values[new_key]['pid'].extend(
                                value_details['pid'])
                            if new_key == key:
                                keys_missing.append(new_key)
                        else:
                            new_values[key] = value_details
                    product['target_scores']['Ports'] = new_values
                    normalized_attributes[category].append("Ports")

                if attribute == "Processor Core" and "Numeric Standardization" in normalization_params:
                    new_values = {}
                    for key, value_details in values.items():
                        if key != "n/a":
                            new_key = normalize_core_number(key)
                            if new_key not in new_values:
                                new_values[new_key] = {
                                    'pid': [], 'score': value_details['score']}
                            new_values[new_key]['pid'].extend(
                                value_details['pid'])
                            if new_key == key:
                                keys_missing.append(new_key)
                        else:
                            new_values[key] = value_details
                    product['target_scores']['Processor Core'] = new_values
                    normalized_attributes[category].append("Processor Core")

                # Handling 'Processor Type' attribute
                if attribute == "Processor Type" and "Name Generalisation" in normalization_params:
                    new_values = {}
                    for key, value_details in values.items():
                        if key != "n/a":
                            new_key = recategorize_product_type(
                                key, processors_dict)
                            if new_key not in new_values:
                                new_values[new_key] = {
                                    'pid': [], 'score': value_details['score']}
                            new_values[new_key]['pid'].extend(
                                value_details['pid'])
                            if new_key == key:
                                keys_missing.append(new_key)
                        else:
                            new_values[key] = value_details
                    product['target_scores']['Processor Type'] = new_values
                    normalized_attributes[category].append("Processor Type")

                if attribute == "Rotational Speed" and "Numeric Standardization" in normalization_params:
                    new_values = {}
                    for key, value_details in values.items():
                        if key != "n/a":
                            new_key = convert_rotational_speed_to_numeric(key)
                            if new_key not in new_values:
                                new_values[new_key] = {
                                    'pid': [], 'score': value_details['score']}
                            new_values[new_key]['pid'].extend(
                                value_details['pid'])
                            if new_key == key:
                                keys_missing.append(new_key)
                        else:
                            new_values[key] = value_details
                    product['target_scores']['Rotational Speed'] = new_values
                    normalized_attributes[category].append("Rotational Speed")

        if category == "Jewelry":
            for attribute, values in product['target_scores'].items():
                #        if attribute == "Gender" and "Binary Classification" in normalization_params:
                #            new_values = {}
                #            for value_key, value_details in values.items():
                #                if value_key != "n/a":
                #                    if value_key in ["her", "Women's", "she", "Ladies"]:
                #                        new_key = "0"
                #                    elif value_key in ["Gents", "Men's"]:
                #                        new_key = "1"
                #                    else:
                #                        continue

                #                    if new_key not in new_values:
                #                        new_values[new_key] = {'pid': [], 'score': 1}

                #                    new_values[new_key]['pid'].extend(value_details['pid'])
                #                else:
                #                    new_values[value_key] = value_details

                #            product['target_scores']['Gender'] = new_values
                #            normalized_attributes[category].append("Gender")

                if attribute == "Product Type" and "Product Type Generalisation" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            new_key = recategorize_product_type(
                                value_key, product_categories_jewelry)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}

                            new_values[new_key]['pid'].extend(
                                value_details['pid'])

                            if new_key == value_key:
                                keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    product['target_scores']['Product Type'] = new_values
                    normalized_attributes[category].append("Product Type")

                if attribute == "Model Number" and "Delete Marks" in normalization_params:
                    new_values = {}
                    for key, value_details in values.items():
                        if key != "n/a":
                            new_key = delete_marks(key)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}
                            new_values[new_key]['pid'].extend(
                                value_details['pid'])
                            if new_key == key:
                                keys_missing.append(new_key)
                        else:
                            new_values[key] = value_details
                    product['target_scores']['Model Number'] = new_values
                    normalized_attributes[category].append('Model Number')

                if attribute == "Brand" and "To Uppercase" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            new_key = value_key.upper()
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}

                            new_values[new_key]['pid'].extend(
                                value_details['pid'])

                            if new_key == value_key:
                                keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    product['target_scores']['Brand'] = new_values
                    normalized_attributes[category].append("Brand")

        if category == "Grocery And Gourmet Food":
            for attribute, values in product['target_scores'].items():
                if attribute == "Product Type" and "Product Type Generalisation" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            new_key = recategorize_product_type(
                                value_key, supermarket_aisles_dict)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}

                            new_values[new_key]['pid'].extend(
                                value_details['pid'])

                            if new_key == value_key:
                                keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    product['target_scores']['Product Type'] = new_values
                    normalized_attributes[category].append("Product Type")

                if attribute == "Size/Weight" and "Unit Conversion" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            parts = value_key.split(';')
                            for part in parts:
                                new_key = str(convert_to_grams(part.lstrip()))
                                if new_key not in new_values:
                                    new_values[new_key] = {
                                        'pid': [], 'score': 1}

                                new_values[new_key]['pid'].extend(
                                    value_details['pid'])

                                if new_key == value_key:
                                    keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    product['target_scores']['Size/Weight'] = new_values
                    normalized_attributes[category].append("Size/Weight")

                if attribute == "Retail UPC" and "Substring Extraction" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            new_key = retail_upc_to_manufacturer_code(
                                value_key)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}
                                new_values[new_key]['pid'].extend(
                                    value_details['pid'])

                                if new_key == value_key:
                                    keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    product['target_scores']['Retail UPC'] = new_values
                    normalized_attributes[category].append("Retail UPC")

                if attribute == "Pack Quantity" and "Numeric Standardization" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            new_keys = extract_digit_from_pack_quantity(
                                value_key)
                            for new_key in new_keys:
                                if new_key not in new_values:
                                    new_values[new_key] = {
                                        'pid': [], 'score': 1}

                                new_values[new_key]['pid'].extend(
                                    value_details['pid'])

                                if new_key == value_key:
                                    keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    product['target_scores']['Pack Quantity'] = new_values
                    normalized_attributes[category].append("Pack Quantity")

                if attribute == "Brand" and "To Uppercase" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            new_key = value_key.upper()
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}

                            new_values[new_key]['pid'].extend(
                                value_details['pid'])

                            if new_key == value_key:
                                keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    product['target_scores']['Brand'] = new_values
                    normalized_attributes[category].append("Brand")

        if category == "Home And Garden":
            for attribute, values in product['target_scores'].items():
                if attribute in ["Height", "Width", "Depth", "Length"] and "Unit Conversion" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            new_key = convert_to_cm(value_key)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}

                            new_values[new_key]['pid'].extend(
                                value_details['pid'])

                            if new_key == value_key:
                                keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    product['target_scores'][attribute] = new_values
                    normalized_attributes[category].append(attribute)

                if attribute == "Manufacturer Stock Number" and "Delete Marks" in normalization_params:
                    new_values = {}
                    for key, value_details in values.items():
                        if key != "n/a":
                            new_key = delete_marks(key)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}
                            new_values[new_key]['pid'].extend(
                                value_details['pid'])
                            if new_key == key:
                                keys_missing.append(new_key)
                        else:
                            new_values[key] = value_details
                    product['target_scores']['Manufacturer Stock Number'] = new_values
                    normalized_attributes[category].append(
                        'Manufacturer Stock Number')

                if attribute == "Color" and "Color Generalization" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            new_keys = recategorize_product_type_multiple(
                                value_key, color_mapping)
                            for new_key in new_keys:
                                if new_key not in new_values:
                                    new_values[new_key] = {
                                        'pid': [], 'score': 1}

                                new_values[new_key]['pid'].extend(
                                    value_details['pid'])

                                if new_key == value_key:
                                    keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    product['target_scores']['Color'] = new_values
                    normalized_attributes[category].append("Color")

                if attribute == "Product Type" and "Product Type Generalisation" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            new_key = recategorize_product_type(
                                value_key, home_and_garden_dict)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}

                            new_values[new_key]['pid'].extend(
                                value_details['pid'])

                            if new_key == value_key:
                                keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    product['target_scores']['Product Type'] = new_values
                    normalized_attributes[category].append('Product Type')

                if attribute == "Retail UPC" and "Substring Extraction" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            new_key = retail_upc_to_manufacturer_code(
                                value_key)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}
                                new_values[new_key]['pid'].extend(
                                    value_details['pid'])

                                if new_key == value_key:
                                    keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    product['target_scores']['Retail UPC'] = new_values
                    normalized_attributes[category].append("Retail UPC")

        if category == "Office Products":
            for attribute, values in product['target_scores'].items():
                if attribute in ["Height", "Width", "Depth", "Length"] and "Unit Conversion" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            new_key = convert_to_cm(value_key)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}

                            new_values[new_key]['pid'].extend(
                                value_details['pid'])

                            if new_key == value_key:
                                keys_missing.append(new_key)

                        else:
                            new_values[value_key] = value_details

                    product['target_scores'][attribute] = new_values
                    normalized_attributes[category].append(attribute)

                if attribute == "Manufacturer Stock Number" and "Delete Marks" in normalization_params:
                    new_values = {}
                    for key, value_details in values.items():
                        if key != "n/a":
                            new_key = delete_marks(key)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}
                            new_values[new_key]['pid'].extend(
                                value_details['pid'])
                            if new_key == key:
                                keys_missing.append(new_key)
                        else:
                            new_values[key] = value_details
                    product['target_scores']['Manufacturer Stock Number'] = new_values
                    normalized_attributes[category].append(
                        'Manufacturer Stock Number')

                if attribute == "Color(s)" and "Color Generalization" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            new_keys = recategorize_product_type_multiple(
                                value_key, color_mapping)
                            for new_key in new_keys:
                                if new_key not in new_values:
                                    new_values[new_key] = {
                                        'pid': [], 'score': 1}

                                new_values[new_key]['pid'].extend(
                                    value_details['pid'])

                                if new_key == value_key:
                                    keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    # Update the Product Type values with the recategorized keys
                    product['target_scores']['Color(s)'] = new_values
                    normalized_attributes[category].append("Color(s)")

                if attribute == "Paper Weight" and "Unit Conversion" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            # print(value_key)
                            new_key = convert_to_kg(value_key)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}

                            new_values[new_key]['pid'].extend(
                                value_details['pid'])

                            if new_key == value_key:
                                keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    # Update the Product Type values with the recategorized keys
                    product['target_scores']["Paper Weight"] = new_values
                    normalized_attributes[category].append("Paper Weight")

                if attribute == "Product Type" and "Product Type Generalisation" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            new_key = recategorize_product_type(
                                value_key, office_products_dict)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}

                            new_values[new_key]['pid'].extend(
                                value_details['pid'])

                            if new_key == value_key:
                                keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    # Update the Product Type values with the recategorized keys
                    product['target_scores']['Product Type'] = new_values
                    normalized_attributes[category].append('Product Type')

                if attribute == "Retail UPC" and "Substring Extraction" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            new_key = retail_upc_to_manufacturer_code(
                                value_key)
                            if new_key not in new_values:
                                new_values[new_key] = {'pid': [], 'score': 1}
                                new_values[new_key]['pid'].extend(
                                    value_details['pid'])

                                if new_key == value_key:
                                    keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    product['target_scores']['Retail UPC'] = new_values
                    normalized_attributes[category].append('Retail UPC')

                if attribute == "Pack Quantity" and "Numeric Standardization" in normalization_params:
                    new_values = {}
                    for value_key, value_details in values.items():
                        if value_key != "n/a":
                            new_keys = extract_digit_from_pack_quantity(
                                value_key)
                            for new_key in new_keys:
                                if new_key not in new_values:
                                    new_values[new_key] = {
                                        'pid': [], 'score': 1}

                                new_values[new_key]['pid'].extend(
                                    value_details['pid'])

                                if new_key == value_key:
                                    keys_missing.append(new_key)
                        else:
                            new_values[value_key] = value_details

                    product['target_scores']['Pack Quantity'] = new_values
                    normalized_attributes[category].append('Pack Quantity')

    for key, values in normalized_attributes.items():
        normalized_attributes[key] = list(set(values))

    for product in data:
        for attribute, values in product['target_scores'].items():
            for value_key, value_details in values.items():
                if isinstance(value_details, dict) and 'pid' in value_details:
                    unique_pid_list = list(set(value_details['pid']))
                    unique_pid_list.sort()
                    value_details['pid'] = unique_pid_list

    # Remove empty categories
    normalized_attributes = {k: v for k,
                             v in normalized_attributes.items() if v}

    #for product in data:
    #    category = product['category']
    #    category_specific_attributes = normalized_attributes.get(category, {})
    #    filtered_target_scores = {
    #        attribute: values for attribute, values in product['target_scores'].items()
    #        if attribute in category_specific_attributes
    #    }
    #    product['target_scores'] = filtered_target_scores

    # Save data in new directory
    params = "_".join(normalization_params) if isinstance(
        normalization_params, list) else normalization_params
    
    name = f"normalized_{split}_{params}"

    with open(os.path.join("data/processed_datasets/old_structure/wdc/normalized/", f"{name}.jsonl"), "w", encoding='utf-8') as f:
        for example in data:
            f.write(json.dumps(example) + "\n")

    name = f"normalized_{split}"
    with open(os.path.join("data/processed_datasets/old_structure/wdc/normalized/", f"{name}.jsonl"), "w", encoding='utf-8') as f:
        for example in data:
            f.write(json.dumps(example) + "\n")




    return normalized_attributes
