#!/usr/bin/env python

#python system imports
import argparse
import copy
import math
import numpy as np
import os
import queue
import select
import socket
import sys
import threading
import time
import traceback

#blender imports
import bmesh
import bpy
import bpy_extras
import mathutils

sys.path.append(os.path.dirname(__file__))
#local imports
import settings


# Credit to effbot.org/librarybook/code.htm for loading variables into current namespace
def keyboard(banner=None):
    import code, sys

    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back

    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)

    code.interact(banner=banner, local=namespace)


context = copy.copy(globals())

class CmdRunner:
    def __init__(self, cmd, context):
        self.context = context
        self.cmd = cmd.replace("#NEWLINE#", "\n")
        self.state = "running"
        
    def run_cmd(self):
        if self.cmd.startswith("eval"):
            f = eval
            fname = 'eval'
            keep_res = True
        elif self.cmd.startswith("exec"):
            f = exec
            fname = 'exec'
            keep_res = False
        else:
            self.state = "error:Command must begin with 'eval:' or 'exec:'"
            return
        
        self.cmd = self.cmd[(self.cmd.index(":")+1):]
        code = compile(self.cmd, '<string>', fname)
        try:
            res = f(code, self.context)
        except:
            self.state = "error:" + traceback.format_exc()
            return
        
        if keep_res:
            res = str(res)
        else:
            res = ""
        self.state = "success%d:%s" % (len(res), res) 
        
    def getState(self):
        return self.state
        
    def done(self):
        return self.state != "running"
        
def cleanup(read_list):
    for s in read_list:
        try:
            s.close()
        except:
            pass

exit_recieved = False
def network_loop(cmdq, port):
    global context
    global exit_recieved
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print("Listening on port %d." % port)
    server_socket.bind((settings.TCP_IP, port))
    server_socket.listen(5)

    read_list = [server_socket]
    cmds = dict()
    while True:
        readable, writable, errored = select.select(read_list, [], [], 0.1)
        for s in readable:
            if s is server_socket:
                client_socket, address = server_socket.accept()
                read_list.append(client_socket)
                print("Connection from" + str(address))
            else:
                data = s.recv(settings.BUFFER_SIZE)
                if data:
                    try:
                        cmd = str(data)
                        cmd = cmd[2:-1]
                        
                        if cmd == "exit":
                            exit_recieved = True
                            cleanup(read_list)
                            return
                        
                        should_got = int(cmd[4:cmd.index(":")])
                        got = len(cmd[(cmd.index(":")+1):])
                        if got < should_got:
                            m = str(s.recv(should_got - got))
                            cmd += m[2:-1]
                        cmds[s] = CmdRunner(cmd, context)
                        cmdq.put(cmds[s])
                    except:
                        print("MALFORMED COMMAND:")
                        print(data)
                else:
                    s.close()
                    read_list.remove(s)
        for key in list(cmds.keys()):
            s = cmds[key].getState()
            key.send(s.encode())
            if s != "running":
                del cmds[key]

def cmd_loop(cmdq):
    global exit_recieved
    while not exit_recieved:
        if cmdq.empty():
            time.sleep(0.001)
        else:
            cmd = cmdq.get()
            cmd.run_cmd()
            
def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--threaded', action="store_true", dest="threaded", default=False)
    parser.add_argument('--port', action="store", dest="port", default=settings.TCP_PORT, type=int)
    args = parser.parse_args(argv)
    
    cmdq = queue.Queue()
    
    net_thread = threading.Thread(target=network_loop, args=(cmdq,args.port,))
    net_thread.daemon = True
    net_thread.start()
    
    if args.threaded:
        thread = threading.Thread(target=cmd_loop, args=(cmdq,))
        thread.daemon = True
        thread.start()
    else:
        s = """
bpy.ops.wm.open_mainfile(filepath='/home/robolab/temp/test.blend')
bpy.context.scene.objects.active = bpy.context.scene.objects['FluidDomain']
bpy.ops.fluid.bake()
"""
        #exec(compile(s, '<string>', 'exec'))
        cmd_loop(cmdq)
        net_thread.join()
    

if __name__ == '__main__':
    main()























