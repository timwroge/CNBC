# Systems Neuroscience  

## Neural Constraints on Learning

### Introduction

Learning is the process of creating new activity patterns with groups of
neurons. Are some of these activity patterns easier to learn? Are there
limits to what a certain population of neurons are capable of
exhibiting? If so, what is the nature of these constraints? The
experiment involved a Rhesus macaques (Macaca mullatta) controlling a
computer cursor by changing neural activity in the primary motor cortex.
A brain computer interface (BCI) was used to record neural activity and
"specify and alter how neural activity mapped to cursor velocity." The
neural activity can be translated to a high dimensional space, the
’Neural space.’ In this space, each dimension represents the activity
of a single neuron. ’Activity patterns can create a low dimensional
subspace, termed the intrinsic manifold.’ This intrinsic manifold shows
the constraints imposed by the current neurocircuitry.

The paper shows that if activity patterns lie within this intrinsic
manifold, then they are easily learned and the monkey will be able to
’proficiently move the cursor’ around the screen. The activity
patterns that outside this manifold, can be shown to not be learnable
(i. e. after hours of practice, this activity cannot be learned). This
gives an explanation to why certain activity patterns in the brain are
easy to learn if it is a task that is similar to what has already been
done, and why other tasks are harder to learn if the brain does not have
experience in this domain.

The brain computer interface was used because all the activation of
neurons could be directly monitored and only these neurons lead to the
activity shown in the experiment.

**What is the Within Manifold Perturbation (WMP)? / Outside manifold
perturbation (OMP)?**

A within manifold perturbation is when the intuitive mapping is
perturbed to lie within the intrinsic manifold and an outside manifold
perturbation is when the intuitive mapping lie outside the manifold. The
WMP maintains the relationship of co-modulation patterns and neural
activity but changes the way that neuronal activity is mapped to the
control space and by extension, cursor kinematics. The OMP does not
maintain the relationship between the intrinsic co-modulation patterns
of neurons but also changes cursor kinematics. The performance will be
impaired by each perturbation but the question is whether or not the new
activity patterns are learnable. Two Rhesus macaques were trained in the
experiment to move the cursor by changing the neuronal activity of 81-91
neurons. The neural population activity is represented in the high
dimensional neural space. In each time step, neural activity is
transferred to the control (or action) space. This is a projected 
from the intrinsic manifold to cursor kinematics using a
Kalman filter.

At the start of the day, the ’intuitive mapping’ was established by
using a BCI mapping that the monkey used to control the cursor easily.
The neurons were also analyzed to see if they are ’co-modulated’ (i.e.
how they fire with the others).

Because of the different nature of the connections of neurons and the
co-modulation, the neuronal activity does not uniformly distribute the
neural space. Due to this fact, an intrinsic manifold can be described
that captures the neural activity in a lower dimensional space. The
intuitive mapping lies within this manifold.

What happens when the control space is within or outside this intrinsic
manifold?

The learning within the manifold is shown to be learnable within a short
amount of time while outside manifold perturbations are shown to not be
learnable.

After the perturbation, the learning is immediately impaired, but in the
within manifold perturbation, the monkey recovers, which is not shown in
the outside manifold perturbation.

The BCI mapping first mapped the neural population activity to the
intrinsic manifold using factor analysis, then the cursor kinematics
were derived from the intrinsic manifold using a Kalman filter.

**Some Alternative explanations**  
The manifold perturbations were checked to see how far off from the
intuitive mapping they lied (cosine distance).

The WMP and OMP were shown to be within the same distance from the
intuitive mapping.

Next, they checked to see if the neurons were seen to be in the same
direction, (i.e. the new mapping was in the preferred direction for each
of the neurons. How is the new control space perturbation defined?

How is the performance/progress defined?

## Methods

The neurons were recorded from the proximal arm region within the
primary motor cortex using 96 channel microelectrode while the monkeys
were in a chair with their head fixed.

The spike threshold was set to 3 times the root mean square voltage of
the baseline neural activity of a monkey in a dark room.

### Blocks

The calibration block completed each day, the intuitive mapping was
established. This mapping was used for 400 trials to establish some
baselines. The mapping was then switched to the perturbed mapping and
the task was completed again. A pertubation session was the combination
of the perturbation and washout blocks.

### Experimental sessions

78 sessions were completed which were composed of 30 within manifold
perturbations and 48 outside manifold perturbations. The session was
thrown out if fewer than 100 trials with the perturbed mapping were
attempted.

The cursor appeared on the screen for 300 ms. A juice reward was given
for each successfully completed task.

# Methods

## Peristimulus Time Histogram (PSTH)

A PSTH is simply a histogram of neurons firing during a specific
interval. This is useful to see how a set of neurons react to a
stimulus.

### Method

**Data:** Time sequential data ordered by time step

**For:** Beginning until End of analysis  
bin the data into equal time interval size bins

**End For**

## Tuning Curves

A tuning curves shows the firing rate as a response to a given stimulus
and is designed to show how a set of neurons can respond to directional
action. This can give insight into the preferred direction of the
neurons firing.

### Cosine Tuning Curves

Cosine tuning curves are a regression technique where you fit the two
parameters, the baseline- \(b _ i\) and the modulation depth \(k _ i\).
\[f _ i = b_ i + k _ i cos ( \theta _ {p, i} - \theta _ m )
    \label{eq:firingrates}\] \[p _ i = \left[ \begin{array}{c}
                    cos( \theta _ {p, i}) \\ 
                    sin( \theta _{p, i} ) 
                \end{array}
        \right]
        \label{eq:coodinates}\]

As shown in Equation [\[eq:firingrates\]](#eq:firingrates), \(f _ i\) is
the firing rates for a given time step. In order to solve this equation,
just switch the coordinate system to that shown in Equation
[\[eq:coodinates\]](#eq:coodinates).

The MATLAB program, `pmdDataSetup` should output the movement angle,
\(\theta _ m\) and the firing rates, \(f _ i\).

# Dimensionality Reduction of Neural Data

First use the aforementioned MATLAB program `pmdDataSetup` to get the
angle and firing rates for each of the neurons, then find the mean and
the covariance of the samples (See
[\[sec:factoranalysis\]](#sec:factoranalysis) for more information about
factor analysis).

The motivation for this work is to help aid in spike sorting, where the
data is intrinsically high dimensional and you would wish to gain some
more insight into the properties of this data.

A simple mathematical representation of neural data would be to define
it as follows: \[x _ n = \alpha _ n 
     \left[ \begin{array}{c}
                v _ 1\\
                v _ 2 \\
                \vdots
     \end{array}
          \right]
     + \beta_n  \left[ \begin{array}{c}
                v _ 1\\
                v _ 2 \\
                \vdots
     \end{array}
     \right]\]

Where \(V\) is the canonical waveform (wave thing), \(\alpha _ n\) is
the amplitude and \(\beta _ n\) is a baseline offset.

If this representation of the neurons are correct, then the degrees of
freedom of the neurons is two. In this case, only \(\alpha _ n\) and
\(\beta _n\) would change.

This would imply that the **intrinsic dimensionality** of the subspace
of the neural firing data would be two. Imagine a plot with the two axes
are defined by \(\alpha _ n\) and \(\beta _ n\), then two separate
neurons would be distinct and each
