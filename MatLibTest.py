__author__ = 'goe'
from pylab import *
import numpy as np

axes([0.1, 0.15, 0.8, 0.75])
plot(np.random.permutation(10))

horizontalLineInValue = 5
plot([0,10],[horizontalLineInValue,horizontalLineInValue])

title('Title', fontsize=20)
# shorthand is also supported and curly's are optional
xlabel(r"""$\"o\ddot o \'e\`e\~n\.x\^y$""", fontsize=20)


show()