nn(upper_legs_net_1, [X], Y, [yes, no]) :: move_upper_legs_1(X,Y).
nn(upper_arms_net_1, [X], Y, [yes, no]) :: move_upper_arms_1(X,Y).
nn(upper_legs_net_2, [X], Y, [yes, no]) :: move_upper_legs_2(X,Y).
nn(upper_arms_net_2, [X], Y, [yes, no]) :: move_upper_arms_2(X,Y).
nn(upper_legs_net_3, [X], Y, [yes, no]) :: move_upper_legs_3(X,Y).
nn(upper_arms_net_3, [X], Y, [yes, no]) :: move_upper_arms_3(X,Y).

activity(X,squat)  :- move_upper_arms_1(X,yes), move_upper_legs_3(X,yes).
activity(X,walk) :- move_upper_legs_1(X,yes), move_upper_legs_2(X,yes), move_upper_legs_3(X,yes).
activity(X,wipe)  :- move_upper_arms_1(X,yes), move_upper_arms_2(X,yes), move_upper_arms_3(X,yes).
