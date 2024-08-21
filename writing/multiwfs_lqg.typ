#import "template.typ": *

#show: project.with(
  title: "Multi-WFS LQG control: mathematical concept",
  authors: (
    "Aditya Sengupta",
  ),
)

For simplicity here we'll say we only look at one mode, and the dynamics of that one mode are dependent only on the current state; this is easily scaled up to multiple modes and to history vectors. This means I'll sometimes write down dynamics matrices that in the case I'm talking about here should just be scalars. 

Let our state be 

$ arrow(x) = (x_"turb", x_"pupil", x_"focal", u) $

where $x_"turb"$ represents the component of turbulence seen by both the pupil-plane and the focal-plane WFS. $x_"pupil"$ and $x_"focal"$ are components seen by just one WFS; $x_"focal"$ encodes what I've been referring to as the NCPA. $u$ is the DM command.

Suppose we have some dynamics models for each of the first three terms. We put this into a block-diagonal state-to-state translation matrix:

$ A = mat(
    A_"turb", 0, 0, 0;
    0, A_"pupil", 0, 0;
    0, 0, A_"focal", 0;
    0, 0, 0, 1
) $

For this to work in reality, we'd need system identification for each of these sub-matrices, i.e. some description of the dynamics of turbulence, of the pupil-plane-only term, and of the focal-plane-only term. 

Together with the input-to-state matrix $B = (0, 0, 0, 1)^T$ and a process noise covariance matrix $V$ with the same dimensions as $A$ and the same block-diagonal structure, we have a dynamics model,

$ x_(n+1) = A x_n + B u_n + cal(N)(0, V). $

We observe two outputs, one per WFS:

$ y_n = C x_n + cal(N)(0, W) = mat(1, 1, 0, -1; 1, 0, 1, -1) x_n + cal(N)(0, W) $

for some measurement noise covariance matrix $W$. $y_n$ is a two-element vector, with one measurement per WFS.

We want to choose $u_n$ such that the absolute value of the noiseless focal-plane WFS reading is minimized. This corresponds to a cost function that is encoded in the LQG formulation in the matrix $Q$, where the scalar cost is $x^T Q x$. In our case, we have

$ Q = C^T mat(0, 0; 0, 1) C = mat(1, 0, 1, -1; 0, 0, 0, 0; 1, 0, 1, -1; -1, 0, -1, 1) $

which under the matrix inner product gives us a minimization objective of $(x_"turb" + x_"focal" - u)^2$, which makes sense. The $x_"pupil"$ term only shows up for state estimation and not optimal control; it remains to choose physically reasonable dynamics models and corresponding appropriate $A, V, W$ for our setting.