# Stage 2 - Preprocessing

### Namespace
The output of each rule which is to be used by the next rule needs to be named
'stage02_preprocessing/<rule_name>/<rule_name>.<neo_format>'.


Each block transforms the AnalogSignal of the input neo file.
The block adds to the description of the AnalogSignal
which operation was performed and the name of the script.
