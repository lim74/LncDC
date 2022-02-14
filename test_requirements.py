import unittest
import pkg_resources
import os
import sys

system = sys.version_info
if system[0] < 3 or system[1] < 9:
    sys.stderr.write("ERROR: Your python version is: "+ str(system[0]) + "." + str(system[1]) +"\n" + "LncDC requires python 3.9 or newer!\n" )
    sys.exit(1)


filename = "requirements.txt"

class Test_requirements(unittest.TestCase):
    
    def test(self):
        file = open(filename)
        requirements = pkg_resources.parse_requirements(file)
        for requirement in requirements:
            requirement=str(requirement)
            with self.subTest(requirement=requirement):
                pkg_resources.require(requirement)
        file.close()

if __name__ == '__main__':
    unittest.main()
