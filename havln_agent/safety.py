"""One common next-step decision for generators, controller and final execution."""


def check_action(action, geometry):
    if action=='MOVE_FORWARD':
        check=geometry.assess_step()
        if check['state']!='EXECUTABLE':
            return None,check['reason']
    return action,None
