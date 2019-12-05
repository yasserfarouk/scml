# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.graphviz',
    'sphinx_autodoc_annotation',
    'sphinx_autodoc_typehints',
    'sphinx.ext.intersphinx',
    'sphinx_automodapi.smart_resolver',
    'nb2plots',
    'sphinx.ext.mathjax',
]

intersphinx_mapping = {'python': ('http://docs.python.org/3', None),
                       'numpy': ('http://docs.scipy.org/doc/numpy/', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
                       # 'matplotlib': ('http://matplotlib.sourceforge.net/', None),
                       }


source_suffix = ['.rst', '.md']

source_parsers = {'.md': 'recommonmark.parser.CommonMarkParser'}

master_doc = 'index'
project = 'SCML'
year = '2019'
author = 'Yasser Mohammad'
copyright = '{0}, {1}'.format(year, author)
version = release = '0.1.2'

pygments_style = 'trac'
templates_path = ['.']
extlinks = {
    'issue': ('https://github.com/yasserfarouk/scml/issues/%s', '#'),
    'pr': ('https://github.com/yasserfarouk/scml/pull/%s', 'PR #'),
}
# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = 'sphinx_rtd_theme'

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
   '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = True
napoleon_use_param = True


graphviz_output_format = 'png'
inheritance_node_attrs = dict(shape='rectangle', fontsize=16, height=0.75,
                              color='white') # , style='filled') # dodgerblue1


imgmath_image_format = 'png'


# If false, no module index is generated.
html_domain_indices = True

automodsumm_inherited_members = True

