nn(upper_legs_net, [X], Y, [high, low, no]) :: move_upper_legs(X,Y).
nn(lower_legs_net, [X], Y, [high, low]) :: move_lower_legs(X,Y).
nn(forearms_net, [X], Y, [high, medium, low, no]) :: move_forearms(X,Y).

activity(X,walk)  :- move_lower_legs(X,high), \+move_forearms(X,high).
activity(X,run)   :- move_lower_legs(X,high), move_forearms(X,high).
activity(X,squat) :- move_lower_legs(X,low), move_upper_legs(X,high).
activity(X,jump)  :- move_lower_legs(X,low), move_upper_legs(X,low).
activity(X,wave)  :- move_lower_legs(X,low), move_upper_legs(X,no), move_forearms(X,medium).
activity(X,clap)  :- move_lower_legs(X,low), move_upper_legs(X,no), move_forearms(X,low).
activity(X,wipe)  :- move_lower_legs(X,low), move_upper_legs(X,no), move_forearms(X,no).
