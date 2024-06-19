<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Symfony\Component\Process\Process;
use Symfony\Component\Process\Exception\ProcessFailedException;

class PythonController extends Controller
{
    // public function index()
    // {
    //     $tempat = base_path('kode3.py');

    //     $proses = new Process([
    //         'python3',$tempat
    //     ]);

    //     $proses->run();

    //     if(!$proses->isSuccessful()){
    //         throw new ProcessFailedException($proses);
    //     }

    //     $output = $proses->getOutput();

    //     return response()->json([
    //         'output'    => $output
    //     ]);
    // }

    public function index()
    {
        // $output = shell_exec('python3'.base_path('/home/user/Project/comvis/simple-app/kode3.py'));
        $output = shell_exec('python3'.base_path('/kode3.py'));

        return response()->json([
            'output'    => $output
        ]);
    }
}
