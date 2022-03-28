===========
isuelogit
===========


.. image:: https://img.shields.io/pypi/v/isuelogit.svg
        :target: https://pypi.python.org/pypi/isuelogit

.. image:: https://img.shields.io/travis/pabloguarda/isuelogit.svg
        :target: https://travis-ci.com/pabloguarda/isuelogit

.. image:: https://readthedocs.org/projects/isuelogit/badge/?version=latest
        :target: https://isuelogit.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/pabloguarda/isuelogit/shield.svg
     :target: https://pyup.io/repos/github/pabloguarda/isuelogit/
     :alt: Updates



Stochastic user equilibrium with logit assignment (`sue-logit`) gives an assignment of travelers in a transportation network where no individual believes that he can unilaterally improve his utility by choosing an alternative paths. A key limitation of `sue-logit` is that it requires to know the parameters of the traveler's utility function beforehand.

The `isuelogit` package addresses this limitation by solving the inverse problem, namely, estimating the parameters of the travelers' utility function using traffic counts, which an output of `sue-logit`.


* Free software: MIT license
* Documentation: https://isuelogit.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
