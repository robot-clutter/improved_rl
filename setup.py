from setuptools import setup

setup(name='bridging_the_gap',
      description='Code for the paper Bridging the gap between learning and heuristic based pushing policies',
      version='0.1',
      author='Iason Sarantopoulos',
      author_email='iasons@auth.gr',
      install_requires=['clt_core @ git+ssh://git@github.com/robot-clutter/clt_core@main#egg=clt_core',
      ]
)