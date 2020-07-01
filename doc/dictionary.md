# Dictionary of main variables

### degrees of freedom (DOF's)

<!-- | <div style="width:100px">variables</div> | description | -->
| <div style="width:120px"></div> | <div style="width:120px"></div>               |
| :------------------------------ |:----------------------------------------------|
| nq                              | position DOF's                                |
| nu                              | velocity DOF's                                |
| nla_g                           | bilateral constraints on position level DOF's |
| nla_gamma                       | bilateral constraints on velocity level DOF's |

### assembler callback

| <div style="width:120px"></div> | <div style="width:120px"></div> |
| :------------------------------ |:--------------------------------|
| assembler_callback              | TODO                            |

## possible subsystem functions

### equations of motion

| <div style="width:120px"></div> | <div style="width:120px"></div> |
| :------------------------------ |:--------------------------------|
| M                               | mass matrix |
| Mu_q                            | TODO |
| f_gyr                           | gyroscopic terms |
| f_gyr_q                         | TODO |
| f_gyr_u                         | TODO |
| f_pot                           | TODO |
| f_pot_q                         | TODO |
| f_npot                          | TODO |
| f_npot_q                        | TODO |
| f_npot_u                        | TODO |
| h                               | TODO |
| h_q                             | TODO |
| h_u                             | TODO |

### kinematic equations

| <div style="width:120px"></div> | <div style="width:120px"></div> |
| :------------------------------ |:--------------------------------|
| q_dot                           | TODO |
| q_dot_q                         | TODO |
| B                               | TODO |
| solver_step_callback            | TODO |

### bilateral constraints on position level

| <div style="width:120px"></div> | <div style="width:120px"></div> |
| :------------------------------ |:--------------------------------|
| g                               | TODO |
| g_q                             | TODO |
| W_g                             | TODO |
| Wla_g_q                         | TODO |

### bilateral constraints on velocity level

| <div style="width:120px"></div> | <div style="width:120px"></div> |
| :------------------------------ |:--------------------------------|
| gamma                           | TODO |
| gamma_q                         | TODO |
| W_gamma                         | TODO |
| W_gammala_gamma_q               | TODO |