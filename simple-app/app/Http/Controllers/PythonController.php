<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Symfony\Component\Process\Process;
use Symfony\Component\Process\Exception\ProcessFailedException;

class PythonController extends Controller
{
    public function index()
    {
        $proses = new Process([
            'python3',base_path('../kode3_yt.py')
        ]);

        if(!$proses->isSuccessful()){
            throw new ProcessFailedException($proses);
        }

        $keluaran = $proses->getOutput();

        return response()->json([
            'keluaran'  => $keluaran
        ]);
    }
}
