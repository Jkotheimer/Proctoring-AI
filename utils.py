def debugTime(message, startTime, finishTime ):
    duration = finishTime - startTime
    duration = round(duration, 4)
    if len(message) > 15:
        message = message[:15]
    else:
        while len(message) < 15:
            message += ' '
    print(' {}| {:0.2f}'.format(message, duration))
    return finishTime
