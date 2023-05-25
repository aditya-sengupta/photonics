#import "template.typ": *

#show: project.with(
  title: "Laboratory experiments with photonic lanterns",
  authors: (
    "Aditya Sengupta",
  ),
)

#set cite(brackets: false)

= What do we want to do with a photonic lantern in the lab?

At the moment, I'm interested in

1. Carrying out identification;
2. Taking an interaction matrix and characterizing the linear range; and
3. Testing linear and quadratic phase reconstruction

= What does it mean to identify a photonic lantern?

We can model lossless optical propagation as a unitary transformation of the input electric field. Photonic lanterns lend themselves to this interpretation well, because their output is an intensity at each single-mode fiber. We can describe the transmission from the pupil plane to the single-mode fiber ports with a matrix $A = U P F$, where $U$ describes the action of the photonic lantern, $P$ is a change of basis into a basis of guided fiber modes (most commonly the LP modes), and $F$ is a pupil-to-focal propagation matrix. 

$P$ and $F$ are relatively well understood and independent of the actual lantern, so for our analysis we can mostly look at the propagation matrix $U$ that takes an $N$-dimensional subspace of the space of focal-plane electric fields and transforms it to a value of the electric field at each of the $N$ ports. We then see the intensities, i.e.

$ p_"out" = |U E_"in"|^2 $

where the $|dot|^2$ operation applies element-wise. In this way, $U$ completely characterizes the behaviour of a photonic lantern.

It's helpful to identify $U$, because knowing the exact behaviour of the PL lets us design algorithms for it a lot more precisely: for example, wavefront reconstruction with a PL. Finite-element simulations are computationally expensive and likely to be less accurate to the particular PL we have, so we need some procedure for finding the elements $U_(j k)$ based on the actual behaviour of the lantern.

_Identification_ means calculating the elements of the lantern's propagation matrix based on empirically-collected data.

= Constraining the matrix as a linear algebra problem

== Problem setup

If we saw $E_"out" = U E_"in"$, this would be an easy problem: for some basis of input electric fields, just apply each one, and the output from basis element $i$ would be the $i$-th column of the propagation matrix. But instead, we only see intensities. Since $U$ and the inputs are complex-valued, this isn't sufficient information to predict the behaviour of the lantern. In addition to this initial set of queries, we'll need to come up with more in order to fully constrain the matrix elements. 

Let's just look at one SMF port; this procedure works independently for all of them because we look at the intensity of each port separately. So we can simplify our problem by considering an $N$-length vector $arrow(s)$ that takes in a vector, say $arrow(v)$, of electric fields and returns the electric field $p = |arrow(s)^T arrow(v)|^2 = |sum_k s_k v_k|^2$ at the SMF output we care about. If we can identify the elements $s_k$ based on the input vectors $arrow(v)$ we choose, we can identify the whole lantern.

== Basis queries

If we sweep over the basis, putting in $arrow(v) = (0, dots, 1, dots, 0)^T$ with the 1 in each position in turn (call these "basis queries"), we'll identify the absolute values of each element: 

$ |s_k|^2 = |(s_1, dots, s_k, dots, s_N) vec(0, dots.v, 1, dots.v, 0)|^2 $

We can write $s_k = |s_k| e^(i phi_k)$. Since we know $|s_i|$, we just have to identify the phase $phi_k$. It's impossible to get absolute phases with just intensity measurements -- we can see this by noticing that for an arbitrary measurement $U arrow(v)$, we can phase-shift the entire matrix by some fixed offset $theta$ and get the same result:

$ |(e^(i theta) dot.circle U) arrow(v)|^2 = | sum_k (e^(i theta) U_(j k)) v_k |^2 = | e^(i theta) sum_k U_(j k)v_k |^2 = e^(-i theta) (U arrow(v))^* (U arrow(v)) e^(i theta) = (U arrow(v))^* (U arrow(v)) = |U arrow(v)|^2 $

so there's no real notion of the "true" phase values that we can measure. 

== Combined queries

What we can measure are phase _differences_ between different $s_k$s, which we can extract using the cosine formula if we put in sums of basis elements. (Call these "combined queries".) If we put in $arrow(v) = (0, dots, underbrace(1, "k"), dots, underbrace(1, "l"), dots, 0)^T$, we'd get $|s_k + s_l|^2$ as our output, which is related to the individual intensities and phases according to

$ |s_k + s_l|^2 = |s_k|^2 + |s_l|^2 + 2 |s_k| |s_l| cos(phi_k - phi_l) $

$ cos(phi_k - phi_l) = (|s_k + s_l|^2 - |s_k|^2 + |s_l|^2) / (2 |s_k| |s_l|). $

This is almost sufficient to recover all the phase differences we're interested in, but since $cos$ is even, we're left with a sign degeneracy; we don't know if we've recovered $phi_k - phi_l$ or $phi_l - phi_k$. This isn't an issue when $N = 2$ because the difference between the two can be thought of as an overall phase shift, of the type that we've established we can ignore. But for higher $N$ we'll recover the "true" phases by accumulating consecutive differences, so it matters that we get all the signs right. 

We can get this information by looking at two combined queries per basis element. Let's say we do combined queries for $(k, l)$, $(k, m)$, and (as part of the next set) $(l, m)$. We can achieve this in practice by saying $l = k + 1$ and $m = k + 2 = l + 1$; in general you look at the difference between each element and the next, and each element and its neighbor two over, since there's no real order on the basis. These queries give us

$ cos(phi_k - phi_l), cos(phi_k - phi_m), cos(phi_l - phi_m). $

These are related according to the cosine formula:

$ cos(phi_k - phi_m) &= cos([phi_k - phi_l] + [phi_l - phi_m]) \ & = cos(phi_k - phi_l) cos(phi_l - phi_m) - sin(phi_k - phi_l) sin(phi_l - phi_m) $

We know all the cosine terms, and from those we know the sine terms up to a sign, so depending on whether you need to fix it or not, we can tell the sign of $phi_l - phi_m$ _relative_ to $phi_k - phi_l$. This is enough information to fully determine the matrix as long as we have a first phase difference.

Let's make this more concrete: suppose we had a 3-port lantern and we wanted to recover the phases $phi_1, phi_2, phi_3$ for each matrix element. Without loss of generality, we can say $phi_1 = 0$, and our measurements give us $cos(phi_2), cos(phi_3),$ and $cos(phi_2 + phi_3)$. If we say $phi_2$ is positive, then $sin(phi_2) > 0$, so all that's left to determine is the sign of $phi_3$, or the sign of $sin(phi_3)$. We can get this from

$ "sign"(sin(phi_3)) = "sign"[(cos(phi_2) cos(phi_3) - cos(phi_2 + phi_3)) / sin(phi_2)] $

and if we had a larger number of ports, we'd be able to get the sign of $phi_4, phi_5$, etc. from the corresponding measurements at higher ports.

Since all of this analysis is for a single row, we'll repeat this $N$ times, or do it once across all the rows as a vector operation.

== Visualizing the queries we need

The figure below shows the combinations of basis elements we'll need to produce at the focal plane. On the $x$ axis, we move across different queries, and on the $y$ axis we track combinations of basis elements; each column is a visual representation of the elements we'll need to use. The basis queries are the first 6, and the combined queries are the remaining 9. In general, we'll need $3N - 3$ propagations to constrain an $N$-port lantern.

#figure(image("../figures/id_procedure.png", width: 100%))

= Choosing bases

So far, we've worked with the assumption that we're able to work in a basis of focal-plane electric field distributions that covers all the aberrations we might be interested in. However, choosing such a basis isn't too easy. The natural basis for a photonic lantern is the LP modes; patterns that describe different solutions to the propagation equation given how large the multi-mode fiber core is relative to the wavelength of light being considered. Unfortunately, it's not too easy to create LP modes at the pupil plane. (fill in the rest tomorrow)

= Incorporating an interaction matrix

The interaction matrix $B$ of a photonic lantern is related to the propagation matrix according to equation 5 from @Lin2022,

$ B_(j k) &= 2 "Im" [A_(j k)^* sum_l A_(j l)] $

which we can simplify under the assumption that we know $A_(j 1)$ (recall that we can safely assume these elements are purely real and have zero phase):

$ B_(j 1) &= 2 [-["Im" A_(j 1)] sum_(l > 1) ["Re" A_(j l)] + ["Re" A_(j 1)] sum_(l > 1) ["Im" A_(j l)]] = 2 ["Re" A_(j 1)] sum_(l > 1) "Im" A_(j l) $

so if we take an interaction matrix in the usual way, applying small-amplitude aberrations and looking at the corresponding outputs, we're able to get a measurement of the accumulated imaginary component across modes for each fiber output. Since the measurements we need for an interaction matrix are just the same as the basis queries we described previously, this isn't a new channel of information. Instead, we can use this as a check on the phases we derive from the combined queries.

In general, the $k$-th column of the interaction matrix will contain the sums of the real and imaginary components of $A_(dots l)$ for every $l eq.not k$, weighted by the negative-imaginary and real components of $A_(dots k)$. In practice, we should only expect to achieve exact equality for the first column, where the basis queries used to derive these are also the measurements we're using for these checks. For the other columns, we'll be making use of phases derived from the combined queries, which will have slightly different noise terms than the basis queries we use for the checks; checking the sums of imaginary components in this way can therefore be useful for checking the amount of accumulated error in the estimation of the complex components $A_(j (2 dots N))$.

The effectiveness of linear control using this interaction matrix depends on how large the linear range turns out to be, which should analytically depend on the magnitude of the second-order correction. Defining the second-order interaction matrix, from @Lin2022 equation 11,

$ C_(j k) = 2 "Re"[A_(j k)^* sum_l A_(j l)]. $

This is likely to have effects that are too small to directly measure, so we likely can't use this to do similar checks as with $B$. However, if we derive this matrix from $A$, we can estimate the linear range. The phase-to-intensity mapping to second order is given by @Lin2022, equation 12:

$ bold(p)_"out" = |A bold(1)|^2 + B bold(Delta phi.alt) - 1/2 C bold(Delta phi.alt)^2 + |A bold(Delta phi.alt)|^2 $

so a coarse measurement of the linear range is the point at which the effects of the $C$ term become significant. For all three matrices, an element $j k$ describes the impact on intensity at the $j$-th output due to a change in the $k$-th mode. So we can look at the linear range in mode $k$ by finding the $bold(Delta phi.alt)_k$ at which the $C$ term is about equal to the $B$ term, i.e.

$ bold(Delta phi.alt)_(k,"cross") = (2B_(j k)) / (C_(j k)). $

When we cross this range of error in mode $k$ for all ports $j$, the assumption of linearity breaks down because we can't sense the first-order effect without being drowned out by the second-order effect.

#bibliography("pl.bib", style: "mla")