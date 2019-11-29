# Stage 03 - Trigger Detection

## Structure
This stage has two kinds of rules.

* Optional transformation rules, which transform the signal (input format = output format), e.g. transformation into a MUA signal. In a workflow one, none, or multiple transformation rule may be applied.

* Trigger detection rules, which determine the times of upwards (and downward) transition and add them as Events to the neo object. Logically, only one trigger detection rule can be applied per workflow.

## Namespace
The output neo structure must contain one Event object per channel with its name equal to the channel number, and the event labels must be either "UP" or "DOWN" depending on the transition type.
