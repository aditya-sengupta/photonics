
def save_telemetry(wait=0):
    warnings.warn("If you see this and you're at Lick, uncommand the line starting with subprocess.run")
    sleep(wait)
    """subprocess.run
        "ssh", "-Y", 
        "user@shimmy.ucolick.org", "/opt/kroot/bin/modify",
        "-s", "saocon", "savedata=1"
    ])
    """