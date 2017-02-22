package com.example.mmota.squeezenet_dse;

import android.content.Context;
import android.content.Intent;
import android.os.Environment;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;
import android.util.Log;
import android.view.WindowManager;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public class SqueezeNet {
    private static final boolean debug = true;
    private static final boolean result = true;
    private static final boolean logEnable = true;
    //private static final int sleepTime = 5000;

    private float[] img;
    private float[] conv1_w;
    private float[] conv1_b;
    private float[] fire2_squeeze1x1_w;
    private float[] fire2_squeeze1x1_b;
    private float[] fire2_expand1x1_w;
    private float[] fire2_expand1x1_b;
    private float[] fire2_expand3x3_w;
    private float[] fire2_expand3x3_b;
    private float[] fire3_squeeze1x1_w;
    private float[] fire3_squeeze1x1_b;
    private float[] fire3_expand1x1_w;
    private float[] fire3_expand1x1_b;
    private float[] fire3_expand3x3_w;
    private float[] fire3_expand3x3_b;
    private float[] fire4_squeeze1x1_w;
    private float[] fire4_squeeze1x1_b;
    private float[] fire4_expand1x1_w;
    private float[] fire4_expand1x1_b;
    private float[] fire4_expand3x3_w;
    private float[] fire4_expand3x3_b;
    private float[] fire5_squeeze1x1_w;
    private float[] fire5_squeeze1x1_b;
    private float[] fire5_expand1x1_w;
    private float[] fire5_expand1x1_b;
    private float[] fire5_expand3x3_w;
    private float[] fire5_expand3x3_b;
    private float[] fire6_squeeze1x1_w;
    private float[] fire6_squeeze1x1_b;
    private float[] fire6_expand1x1_w;
    private float[] fire6_expand1x1_b;
    private float[] fire6_expand3x3_w;
    private float[] fire6_expand3x3_b;
    private float[] fire7_squeeze1x1_w;
    private float[] fire7_squeeze1x1_b;
    private float[] fire7_expand1x1_w;
    private float[] fire7_expand1x1_b;
    private float[] fire7_expand3x3_w;
    private float[] fire7_expand3x3_b;
    private float[] fire8_squeeze1x1_w;
    private float[] fire8_squeeze1x1_b;
    private float[] fire8_expand1x1_w;
    private float[] fire8_expand1x1_b;
    private float[] fire8_expand3x3_w;
    private float[] fire8_expand3x3_b;
    private float[] fire9_squeeze1x1_w;
    private float[] fire9_squeeze1x1_b;
    private float[] fire9_expand1x1_w;
    private float[] fire9_expand1x1_b;
    private float[] fire9_expand3x3_w;
    private float[] fire9_expand3x3_b;
    private float[] conv10_w;
    private float[] conv10_b;





    private static final String tag = "SQNET_DSE";

    public void seqSqueezeNet (Context context) throws InterruptedException {
        loadParameters("SqueezeNet/Logs", "sequential.txt", "/SqueezeNet/Parameters/Normal", logEnable);

        //Thread.sleep(sleepTime);

        Intent createDatabase = new Intent("com.quicinc.trepn.start_profiling");
        createDatabase.putExtra("com.quicinc.trepn.database_file", "Sequential");
        context.sendBroadcast(createDatabase);

        Intent startProfiling = new Intent("com.quicinc.trepn.start_profiling");
        context.sendBroadcast(startProfiling);

        float[] conv1 = new float[111 * 111 * 96];
        convRelu(img, conv1, conv1_w, conv1_b, 227, 227, 3, 96, 7, 2, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_Conv1", "Conv1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "conv1.bin", conv1, 111 * 111 * 96);
        Log.i(tag, "Conv1 finished.");

        //Thread.sleep(sleepTime);

        float[] pool1 = new float[55 * 55 * 96];
        maxpool(conv1, pool1, 111, 111, 96, 3, 2, context, "SqueezeNet/Logs", "sequential.txt", "sequential_Pool1", "Pool1 (ms)", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "pool1.bin", pool1, 55 * 55 * 96);
        Log.i(tag, "Pool1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire2_squeeze1x1 = new float[55 * 55 * 16];
        convRelu(pool1, fire2_squeeze1x1, fire2_squeeze1x1_w, fire2_squeeze1x1_b, 55, 55, 96, 16, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire2_squeeze1x1", "Fire2_squeeze1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire2_squeeze1x1.bin", fire2_squeeze1x1, 55 * 55 * 16);
        Log.i(tag, "Fire2_squeeze1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire2_expand1x1 = new float[55 * 55 * 64];
        convRelu(fire2_squeeze1x1, fire2_expand1x1, fire2_expand1x1_w, fire2_expand1x1_b, 55, 55, 16, 64, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire2_expand1x1", "Fire2_expand1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire2_expand1x1.bin", fire2_expand1x1, 55 * 55 * 64);
        Log.i(tag, "Fire2_expand1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire2_expand3x3 = new float[55 * 55 * 64];
        convRelu(fire2_squeeze1x1, fire2_expand3x3, fire2_expand3x3_w, fire2_expand3x3_b, 55, 55, 16, 64, 3, 1, 1, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire2_expand3x3", "Fire2_expand3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire2_expand3x3.bin", fire2_expand3x3, 55 * 55 * 64);
        Log.i(tag, "Fire2_expand3x3 finished.");

        //Thread.sleep(sleepTime);

        float[] fire2_concat =  new float[55 * 55 * 128];
        System.arraycopy(fire2_expand1x1, 0, fire2_concat, 0, 55 * 55 * 64);
        System.arraycopy(fire2_expand3x3, 0, fire2_concat, 55 * 55 * 64, 55 * 55 * 64);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire2_concat.bin", fire2_concat, 55 * 55 * 128);
        Log.i(tag, "Fire2_concat finished.");

        //Thread.sleep(sleepTime);

        float[] fire3_squeeze1x1 =  new float[55 * 55 * 16];
        convRelu(fire2_concat, fire3_squeeze1x1, fire3_squeeze1x1_w, fire3_squeeze1x1_b, 55, 55, 128, 16, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire3_squeeze1x1", "Fire3_squeeze1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire3_squeeze1x1.bin", fire3_squeeze1x1, 55 * 55 * 16);
        Log.i(tag, "Fire3_squeeze1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire3_expand1x1 = new float[55 * 55 * 64];
        convRelu(fire3_squeeze1x1, fire3_expand1x1, fire3_expand1x1_w, fire3_expand1x1_b, 55, 55, 16, 64, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire3_expand1x1", "Fire3_expand1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire3_expand1x1.bin", fire3_expand1x1, 55 * 55 * 64);
        Log.i(tag, "Fire3_expand1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire3_expand3x3 = new float[55 * 55 * 64];
        convRelu(fire3_squeeze1x1, fire3_expand3x3, fire3_expand3x3_w, fire3_expand3x3_b, 55, 55, 16, 64, 3, 1, 1, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire3_expand3x3", "Fire3_expand3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire3_expand3x3.bin", fire3_expand3x3, 55 * 55 * 64);
        Log.i(tag, "Fire3_expand3x3 finished.");

        //Thread.sleep(sleepTime);

        float[] fire3_concat = new float[55 * 55 * 128];
        System.arraycopy(fire3_expand1x1, 0, fire3_concat, 0, 55 * 55 * 64);
        System.arraycopy(fire3_expand3x3, 0, fire3_concat, 55 * 55 * 64, 55 * 55 * 64);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire3_concat.bin", fire3_concat, 55 * 55 * 128);
        Log.i(tag, "Fire3_concat finished.");

        float[] fire4_squeeze1x1 =  new float[55 * 55 * 32];
        convRelu(fire3_concat, fire4_squeeze1x1, fire4_squeeze1x1_w, fire4_squeeze1x1_b, 55, 55, 128, 32, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire4_squeeze1x1", "Fire4_squeeze1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire4_squeeze1x1.bin", fire4_squeeze1x1, 55 * 55 * 32);
        Log.i(tag, "Fire4_squeeze1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire4_expand1x1 = new float[55 * 55 * 128];
        convRelu(fire4_squeeze1x1, fire4_expand1x1, fire4_expand1x1_w, fire4_expand1x1_b, 55, 55, 32, 128, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire4_expand1x1", "Fire4_expand1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire4_expand1x1.bin", fire4_expand1x1, 55 * 55 * 128);
        Log.i(tag, "Fire4_expand1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire4_expand3x3 = new float[55 * 55 * 128];
        convRelu(fire4_squeeze1x1, fire4_expand3x3, fire4_expand3x3_w, fire4_expand3x3_b, 55, 55, 32, 128, 3, 1, 1, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire4_expand3x3", "Fire4_expand3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire4_expand3x3.bin", fire4_expand3x3, 55 * 55 * 128);
        Log.i(tag, "Fire4_expand3x3 finished.");

        //Thread.sleep(sleepTime);

        float[] fire4_concat = new float[55 * 55 * 256];
        System.arraycopy(fire4_expand1x1, 0, fire4_concat, 0, 55 * 55 * 128);
        System.arraycopy(fire4_expand3x3, 0, fire4_concat, 55 * 55 * 128, 55 * 55 * 128);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire4_concat.bin", fire4_concat, 55 * 55 * 256);
        Log.i(tag, "Fire4_concat finished.");

        //Thread.sleep(sleepTime);

        float[] pool4 = new float[27 * 27 * 256];
        maxpool(fire4_concat, pool4, 55, 55, 256, 3, 2, context, "SqueezeNet/Logs", "sequential.txt", "sequential_Pool2", "Pool2 (ms)", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "pool4.bin", pool4, 27 * 27 * 256);
        Log.i(tag, "Pool2 finished.");

        //Thread.sleep(sleepTime);

        float[] fire5_squeeze1x1 =  new float[27 * 27 * 32];
        convRelu(pool4, fire5_squeeze1x1, fire5_squeeze1x1_w, fire5_squeeze1x1_b, 27, 27, 256, 32, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire5_squeeze1x1", "Fire5_squeeze1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire5_squeeze1x1.bin", fire5_squeeze1x1, 27 * 27 * 32);
        Log.i(tag, "Fire5_squeeze1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire5_expand1x1 = new float[27 * 27 * 128];
        convRelu(fire5_squeeze1x1, fire5_expand1x1, fire5_expand1x1_w, fire5_expand1x1_b, 27, 27, 32, 128, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire5_expand1x1", "Fire5_expand1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire5_expand1x1.bin", fire5_expand1x1, 27 * 27 * 128);
        Log.i(tag, "Fire5_expand1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire5_expand3x3 = new float[27 * 27 * 128];
        convRelu(fire5_squeeze1x1, fire5_expand3x3, fire5_expand3x3_w, fire5_expand3x3_b, 27, 27, 32, 128, 3, 1, 1, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire5_expand3x3", "Fire5_expand3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire5_expand3x3.bin", fire5_expand3x3, 27 * 27 * 128);
        Log.i(tag, "Fire5_expand3x3 finished.");

        //Thread.sleep(sleepTime);

        float[] fire5_concat = new float[27 * 27 * 256];
        System.arraycopy(fire5_expand1x1, 0, fire5_concat, 0, 27 * 27 * 128);
        System.arraycopy(fire5_expand3x3, 0, fire5_concat, 27 * 27 * 128, 27 * 27 * 128);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire5_concat.bin", fire5_concat, 27 * 27 * 256);
        Log.i(tag, "Fire5_concat finished.");

        float[] fire6_squeeze1x1 =  new float[27 * 27 * 48];
        convRelu(fire5_concat, fire6_squeeze1x1, fire6_squeeze1x1_w, fire6_squeeze1x1_b, 27, 27, 256, 48, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire6_squeeze1x1", "Fire6_squeeze1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire6_squeeze1x1.bin", fire6_squeeze1x1, 27 * 27 * 48);
        Log.i(tag, "Fire6_squeeze1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire6_expand1x1 = new float[27 * 27 * 192];
        convRelu(fire6_squeeze1x1, fire6_expand1x1, fire6_expand1x1_w, fire6_expand1x1_b, 27, 27, 48, 192, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire6_expand1x1", "Fire6_expand1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire6_expand1x1.bin", fire6_expand1x1, 27 * 27 * 192);
        Log.i(tag, "Fire6_expand1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire6_expand3x3 = new float[27 * 27 * 192];
        convRelu(fire6_squeeze1x1, fire6_expand3x3, fire6_expand3x3_w, fire6_expand3x3_b, 27, 27, 48, 192, 3, 1, 1, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire6_expand3x3", "Fire6_expand3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire6_expand3x3.bin", fire6_expand3x3, 27 * 27 * 192);
        Log.i(tag, "Fire6_expand3x3 finished.");

        //Thread.sleep(sleepTime);

        float[] fire6_concat = new float[27 * 27 * 384];
        System.arraycopy(fire6_expand1x1, 0, fire6_concat, 0, 27 * 27 * 192);
        System.arraycopy(fire6_expand3x3, 0, fire6_concat, 27 * 27 * 192, 27 * 27 * 192);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire6_concat.bin", fire6_concat, 27 * 27 * 384);
        Log.i(tag, "Fire6_concat finished.");

        float[] fire7_squeeze1x1 =  new float[27 * 27 * 48];
        convRelu(fire6_concat, fire7_squeeze1x1, fire7_squeeze1x1_w, fire7_squeeze1x1_b, 27, 27, 384, 48, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire7_squeeze1x1", "Fire7_squeeze1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire7_squeeze1x1.bin", fire7_squeeze1x1, 27 * 27 * 48);
        Log.i(tag, "Fire7_squeeze1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire7_expand1x1 = new float[27 * 27 * 192];
        convRelu(fire7_squeeze1x1, fire7_expand1x1, fire7_expand1x1_w, fire7_expand1x1_b, 27, 27, 48, 192, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire7_expand1x1", "Fire7_expand1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire7_expand1x1.bin", fire7_expand1x1, 27 * 27 * 192);
        Log.i(tag, "Fire7_expand1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire7_expand3x3 = new float[27 * 27 * 192];
        convRelu(fire7_squeeze1x1, fire7_expand3x3, fire7_expand3x3_w, fire7_expand3x3_b, 27, 27, 48, 192, 3, 1, 1, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire7_expand3x3", "Fire7_expand3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire7_expand3x3.bin", fire7_expand3x3, 27 * 27 * 192);
        Log.i(tag, "Fire7_expand3x3 finished.");

        //Thread.sleep(sleepTime);

        float[] fire7_concat = new float[27 * 27 * 384];
        System.arraycopy(fire7_expand1x1, 0, fire7_concat, 0, 27 * 27 * 192);
        System.arraycopy(fire7_expand3x3, 0, fire7_concat, 27 * 27 * 192, 27 * 27 * 192);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire7_concat.bin", fire7_concat, 27 * 27 * 384);
        Log.i(tag, "Fire7_concat finished.");

        float[] fire8_squeeze1x1 =  new float[27 * 27 * 64];
        convRelu(fire7_concat, fire8_squeeze1x1, fire8_squeeze1x1_w, fire8_squeeze1x1_b, 27, 27, 384, 64, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire8_squeeze1x1", "Fire8_squeeze1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire8_squeeze1x1.bin", fire8_squeeze1x1, 27 * 27 * 64);
        Log.i(tag, "Fire8_squeeze1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire8_expand1x1 = new float[27 * 27 * 256];
        convRelu(fire8_squeeze1x1, fire8_expand1x1, fire8_expand1x1_w, fire8_expand1x1_b, 27, 27, 64, 256, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire8_expand1x1", "Fire8_expand1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire8_expand1x1.bin", fire8_expand1x1, 27 * 27 * 256);
        Log.i(tag, "Fire8_expand1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire8_expand3x3 = new float[27 * 27 * 256];
        convRelu(fire8_squeeze1x1, fire8_expand3x3, fire8_expand3x3_w, fire8_expand3x3_b, 27, 27, 64, 256, 3, 1, 1, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire8_expand3x3", "Fire8_expand3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire8_expand3x3.bin", fire8_expand3x3, 27 * 27 * 256);
        Log.i(tag, "Fire8_expand3x3 finished.");

        //Thread.sleep(sleepTime);

        float[] fire8_concat = new float[27 * 27 * 512];
        System.arraycopy(fire8_expand1x1, 0, fire8_concat, 0, 27 * 27 * 256);
        System.arraycopy(fire8_expand3x3, 0, fire8_concat, 27 * 27 * 256, 27 * 27 * 256);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire8_concat.bin", fire8_concat, 27 * 27 * 512);
        Log.i(tag, "Fire8_concat finished.");

        //Thread.sleep(sleepTime);

        float[] pool8 = new float[13 * 13 * 512];
        maxpool(fire8_concat, pool8, 27, 27, 512, 3, 2, context, "SqueezeNet/Logs", "sequential.txt", "sequential_Pool8", "Pool8 (ms)", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "pool8.bin", pool8, 13 * 13 * 512);
        Log.i(tag, "Pool8 finished.");

        //Thread.sleep(sleepTime);

        float[] fire9_squeeze1x1 =  new float[13 * 13 * 64];
        convRelu(pool8, fire9_squeeze1x1, fire9_squeeze1x1_w, fire9_squeeze1x1_b, 13, 13, 512, 64, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire9_squeeze1x1", "Fire9_squeeze1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire9_squeeze1x1.bin", fire9_squeeze1x1, 13 * 13 * 64);
        Log.i(tag, "Fire9_squeeze1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire9_expand1x1 = new float[13 * 13 * 256];
        convRelu(fire9_squeeze1x1, fire9_expand1x1, fire9_expand1x1_w, fire9_expand1x1_b, 13, 13, 64, 256, 1, 1, 0, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire9_expand1x1", "Fire9_expand1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire9_expand1x1.bin", fire9_expand1x1, 13 * 13 * 256);
        Log.i(tag, "Fire9_expand1x1 finished.");

        //Thread.sleep(sleepTime);

        float[] fire9_expand3x3 = new float[13 * 13 * 256];
        convRelu(fire9_squeeze1x1, fire9_expand3x3, fire9_expand3x3_w, fire9_expand3x3_b, 13, 13, 64, 256, 3, 1, 1, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_fire9_expand3x3", "Fire9_expand3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire9_expand3x3.bin", fire9_expand3x3, 13 * 13 * 256);
        Log.i(tag, "Fire9_expand3x3 finished.");

        //Thread.sleep(sleepTime);

        float[] fire9_concat = new float[13 * 13 * 512];
        System.arraycopy(fire9_expand1x1, 0, fire9_concat, 0, 13 * 13 * 256);
        System.arraycopy(fire9_expand3x3, 0, fire9_concat, 13 * 13 * 256, 13 * 13 * 256);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "fire9_concat.bin", fire9_concat, 13 * 13 * 512);
        Log.i(tag, "Fire9_concat finished.");

        //Thread.sleep(sleepTime);

        float[] conv10 = new float[15 * 15 * 1000];
        convRelu(fire9_concat, conv10, conv10_w, conv10_b, 13, 13, 512, 1000, 1, 1, 1, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_Conv10", "Conv10 (ms): ", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "conv10.bin", conv10, 15 * 15 * 1000);
        Log.i(tag, "Conv10 finished.");

        //Thread.sleep(sleepTime);

        float[] pool10 = new float[1000];
        avgpool(conv10, pool10, 15, 15, 1000, 15, 1, context, "SqueezeNet/Logs", "sequential.txt", "sequential_Pool10", "Pool10 (ms)", logEnable);
        if (debug)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "pool10.bin", pool10, 1000);
        Log.i(tag, "Pool10 finished.");

        //Thread.sleep(sleepTime);

        float[] prob = new float[1000];
        softmax(pool10, prob, 1000, "SqueezeNet/Logs", "sequential.txt", "Prob (ms)", logEnable);

        Intent stopProfiling = new Intent("com.quicinc.trepn.stop_profiling");
        context.sendBroadcast(stopProfiling);

        if (result)
            binaryDumper("SqueezeNet/Intermediate_rslts/Sequential", "prob.bin", prob, 1000);
        Log.i(tag, "Prob finished.");

    }
    public void parSqueezeNet (Context context) throws InterruptedException {

        loadParameters("SqueezeNet/Logs", "parallel.txt", "/SqueezeNet/Parameters/Vectorized", logEnable);

        RenderScript rs = RenderScript.create(context);
        ScriptC_convNet convNet = new ScriptC_convNet(rs);

        Type.Builder imgType = new Type.Builder(rs, Element.F32_4(rs)).setX(227 * 227);
        Allocation imgAllocation = Allocation.createTyped(rs, imgType.create());
        paraReshape(rs, convNet, imgAllocation);

        Allocation conv1Allocation = Allocation.createSized(rs, Element.F32(rs), 111 * 111 * 96);
        int[] conv1TypeSet = {1, 2, 4, 6, 8, 12, 24};
        for (int convType : conv1TypeSet) {
            int parallelOFMs = 96 / convType;
            paraConvRelu(rs, convNet, 7 * 7 * 96, 96, conv1_w, conv1_b, 111 * 111 * parallelOFMs, 7,
                    111, 111, 227, 227, 96, 3, 2, 0, 1, 0, imgAllocation, conv1Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "Conv1 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[111 * 111 * 96];
                conv1Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "conv1.bin", tmp, 111 * 111 * 96);
            }
        }
        Log.i(tag, "Conv1 finished.");

        Allocation pool1Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 96);
        paraMaxPool(rs, convNet, 55 * 55 * 24, 3, 111, 111, 2, conv1Allocation, pool1Allocation, "SqueezeNet/Logs", "4.txt", "Pool1 (ms): ", logEnable);
        if (debug) {
            float[] pool1 = new float[55 * 55 * 96];
            pool1Allocation.copyTo(pool1);
            binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/4", "pool1.bin", pool1, 55 * 55 * 96);
        }
        Log.i(tag, "Pool1 finished.");

        Allocation fire2_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 16);
        int[] fire2_squeeze1x1_TypeSet = {1, 2, 4};
        for (int convType : fire2_squeeze1x1_TypeSet) {
            int parallelOFMs = 16 / convType;
            paraConvRelu(rs, convNet, 24 * 16, 16, fire2_squeeze1x1_w, fire2_squeeze1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 16, 96, 1, 0, 24, 0, pool1Allocation, fire2_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire2_squeeze1x1 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[55 * 55 * 16];
                fire2_squeeze1x1_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire2_squeeze1x1.bin", tmp, 55 * 55 * 16);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire2_squeeze1x1 finished.");

        Allocation fire2_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 128);
        int[] fire2_expand_TypeSet = {1, 2, 4, 8, 16};
        for (int convType : fire2_expand_TypeSet) {
            int parallelOFMs = 64 / convType;
            paraConvRelu(rs, convNet, 4 * 64, 64, fire2_expand1x1_w, fire2_expand1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 64, 16, 1, 0, 4, 0, fire2_squeeze1x1_Allocation, fire2_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire2_expand1x1 (ms): ", logEnable);
            //Thread.sleep(sleepTime);
            paraConvRelu(rs, convNet, 3 * 3 * 4 * 64, 64, fire2_expand3x3_w, fire2_expand3x3_b, 55 * 55 * parallelOFMs, 3,
                    55, 55, 55, 55, 64, 16, 1, 1, 4, 55 * 55 * 64, fire2_squeeze1x1_Allocation, fire2_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire2_expand3x3 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[55 * 55 * 128];
                fire2_expand_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire2_concat.bin", tmp, 55 * 55 * 128);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire2_expand finished.");

        Allocation fire3_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 16);
        int[] fire3_squeeze1x1_TypeSet = {1, 2, 4};
        for (int convType : fire3_squeeze1x1_TypeSet) {
            int parallelOFMs = 16 / convType;
            paraConvRelu(rs, convNet, 32 * 16, 16, fire3_squeeze1x1_w, fire3_squeeze1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 16, 128, 1, 0, 32, 0, fire2_expand_Allocation, fire3_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire3_squeeze1x1 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[55 * 55 * 16];
                fire3_squeeze1x1_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire3_squeeze1x1.bin", tmp, 55 * 55 * 16);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire3_squeeze1x1 finished.");

        Allocation fire3_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 128);
        int[] fire3_expand_TypeSet = {1, 2, 4, 8, 16};
        for (int convType : fire3_expand_TypeSet) {
            int parallelOFMs = 64 / convType;
            paraConvRelu(rs, convNet, 4 * 64, 64, fire3_expand1x1_w, fire3_expand1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 64, 16, 1, 0, 4, 0, fire3_squeeze1x1_Allocation, fire3_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire3_expand1x1 (ms): ", logEnable);
            //Thread.sleep(sleepTime);
            paraConvRelu(rs, convNet, 3 * 3 * 4 * 64, 64, fire3_expand3x3_w, fire3_expand3x3_b, 55 * 55 * parallelOFMs, 3,
                    55, 55, 55, 55, 64, 16, 1, 1, 4, 55 * 55 * 64, fire3_squeeze1x1_Allocation, fire3_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire3_expand3x3 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[55 * 55 * 128];
                fire3_expand_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire3_concat.bin", tmp, 55 * 55 * 128);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire3_expand finished.");

        Allocation fire4_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 32);
        int[] fire4_squeeze1x1_TypeSet = {1, 2, 4, 8};
        for (int convType : fire4_squeeze1x1_TypeSet) {
            int parallelOFMs = 32 / convType;
            paraConvRelu(rs, convNet, 32 * 32, 32, fire4_squeeze1x1_w, fire4_squeeze1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 32, 128, 1, 0, 32, 0, fire3_expand_Allocation, fire4_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire4_squeeze1x1 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[55 * 55 * 32];
                fire4_squeeze1x1_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire4_squeeze1x1.bin", tmp, 55 * 55 * 32);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire4_squeeze1x1 finished.");

        Allocation fire4_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 256);
        int[] fire4_expand_TypeSet = {1, 2, 4, 8, 16, 32};
        for (int convType : fire4_expand_TypeSet) {
            int parallelOFMs = 128 / convType;
            paraConvRelu(rs, convNet, 8 * 128, 128, fire4_expand1x1_w, fire4_expand1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 128, 32, 1, 0, 8, 0, fire4_squeeze1x1_Allocation, fire4_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire4_expand1x1 (ms): ", logEnable);
            //Thread.sleep(sleepTime);
            paraConvRelu(rs, convNet, 3 * 3 * 8 * 128, 128, fire4_expand3x3_w, fire4_expand3x3_b, 55 * 55 * parallelOFMs, 3,
                    55, 55, 55, 55, 128, 32, 1, 1, 8, 55 * 55 * 128, fire4_squeeze1x1_Allocation, fire4_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire4_expand3x3 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[55 * 55 * 256];
                fire4_expand_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire4_concat.bin", tmp, 55 * 55 * 256);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire4_expand finished.");

        Allocation pool4Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 256);
        paraMaxPool(rs, convNet, 27 * 27 * 64, 3, 55, 55, 2, fire4_expand_Allocation, pool4Allocation, "SqueezeNet/Logs", "4.txt", "Pool4 (ms): ", logEnable);
        if (debug) {
            float[] tmp = new float[27 * 27 * 256];
            pool4Allocation.copyTo(tmp);
            binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/4", "pool4.bin", tmp, 27 * 27 * 256);
        }
        Log.i(tag, "Pool4 finished.");

        Allocation fire5_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 32);
        int[] fire5_squeeze1x1_TypeSet = {1, 2, 4, 8};
        for (int convType : fire5_squeeze1x1_TypeSet) {
            int parallelOFMs = 32 / convType;
            paraConvRelu(rs, convNet, 64 * 32, 32, fire5_squeeze1x1_w, fire5_squeeze1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 32, 256, 1, 0, 64, 0, pool4Allocation, fire5_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire5_squeeze1x1 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[27 * 27 * 32];
                fire5_squeeze1x1_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire5_squeeze1x1.bin", tmp, 27 * 27 * 32);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire5_squeeze1x1 finished.");

        Allocation fire5_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 256);
        int[] fire5_expand_TypeSet = {1, 2, 4, 8, 16, 32};
        for (int convType : fire5_expand_TypeSet) {
            int parallelOFMs = 128 / convType;
            paraConvRelu(rs, convNet, 8 * 128, 128, fire5_expand1x1_w, fire5_expand1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 128, 32, 1, 0, 8, 0, fire5_squeeze1x1_Allocation, fire5_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire5_expand1x1 (ms): ", logEnable);
            //Thread.sleep(sleepTime);
            paraConvRelu(rs, convNet, 3 * 3 * 8 * 128, 128, fire5_expand3x3_w, fire5_expand3x3_b, 27 * 27 * parallelOFMs, 3,
                    27, 27, 27, 27, 128, 32, 1, 1, 8, 27 * 27 * 128, fire5_squeeze1x1_Allocation, fire5_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire5_expand3x3 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[27 * 27 * 256];
                fire5_expand_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire5_concat.bin", tmp, 27 * 27 * 256);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire5_expand finished.");

        Allocation fire6_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 48);
        int[] fire6_squeeze1x1_TypeSet = {1, 2, 4, 6, 12};
        for (int convType : fire6_squeeze1x1_TypeSet) {
            int parallelOFMs = 48 / convType;
            paraConvRelu(rs, convNet, 64 * 48, 48, fire6_squeeze1x1_w, fire6_squeeze1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 48, 256, 1, 0, 64, 0, fire5_expand_Allocation, fire6_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire6_squeeze1x1 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[27 * 27 * 48];
                fire6_squeeze1x1_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire6_squeeze1x1.bin", tmp, 27 * 27 * 48);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire6_squeeze1x1 finished.");

        Allocation fire6_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 384);
        int[] fire6_expand_TypeSet = {1, 2, 4, 6, 8, 12, 16, 24, 48};
        for (int convType : fire6_expand_TypeSet) {
            int parallelOFMs = 192 / convType;
            paraConvRelu(rs, convNet, 12 * 192, 192, fire6_expand1x1_w, fire6_expand1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 192, 48, 1, 0, 12, 0, fire6_squeeze1x1_Allocation, fire6_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire6_expand1x1 (ms): ", logEnable);
            //Thread.sleep(sleepTime);
            paraConvRelu(rs, convNet, 3 * 3 * 12 * 192, 192, fire6_expand3x3_w, fire6_expand3x3_b, 27 * 27 * parallelOFMs, 3,
                    27, 27, 27, 27, 192, 48, 1, 1, 12, 27 * 27 * 192, fire6_squeeze1x1_Allocation, fire6_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire6_expand3x3 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[27 * 27 * 384];
                fire6_expand_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire6_concat.bin", tmp, 27 * 27 * 384);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire6_expand finished.");

        Allocation fire7_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 48);
        int[] fire7_squeeze1x1_TypeSet = {1, 2, 4, 6, 12};
        for (int convType : fire7_squeeze1x1_TypeSet) {
            int parallelOFMs = 48 / convType;
            paraConvRelu(rs, convNet, 96 * 48, 48, fire7_squeeze1x1_w, fire7_squeeze1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 48, 384, 1, 0, 96, 0, fire6_expand_Allocation, fire7_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire7_squeeze1x1 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[27 * 27 * 48];
                fire7_squeeze1x1_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire7_squeeze1x1.bin", tmp, 27 * 27 * 48);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire7_squeeze1x1 finished.");

        Allocation fire7_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 384);
        int[] fire7_expand_TypeSet = {1, 2, 4, 6, 8, 12, 16, 24, 48};
        for (int convType : fire7_expand_TypeSet) {
            int parallelOFMs = 192 / convType;
            paraConvRelu(rs, convNet, 12 * 192, 192, fire7_expand1x1_w, fire7_expand1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 192, 48, 1, 0, 12, 0, fire7_squeeze1x1_Allocation, fire7_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire7_expand1x1 (ms): ", logEnable);
            //Thread.sleep(sleepTime);
            paraConvRelu(rs, convNet, 3 * 3 * 12 * 192, 192, fire7_expand3x3_w, fire7_expand3x3_b, 27 * 27 * parallelOFMs, 3,
                    27, 27, 27, 27, 192, 48, 1, 1, 12, 27 * 27 * 192, fire7_squeeze1x1_Allocation, fire7_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire7_expand3x3 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[27 * 27 * 384];
                fire7_expand_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire7_concat.bin", tmp, 27 * 27 * 384);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire7_expand finished.");

        Allocation fire8_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 64);
        int[] fire8_squeeze1x1_TypeSet = {1, 2, 4, 8, 16};
        for (int convType : fire8_squeeze1x1_TypeSet) {
            int parallelOFMs = 64 / convType;
            paraConvRelu(rs, convNet, 96 * 64, 64, fire8_squeeze1x1_w, fire8_squeeze1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 64, 384, 1, 0, 96, 0, fire7_expand_Allocation, fire8_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire8_squeeze1x1 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[27 * 27 * 64];
                fire8_squeeze1x1_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire8_squeeze1x1.bin", tmp, 27 * 27 * 64);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire8_squeeze1x1 finished.");

        Allocation fire8_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 512);
        int[] fire8_expand_TypeSet = {1, 2, 4, 8, 16, 32, 64};
        for (int convType : fire8_expand_TypeSet) {
            int parallelOFMs = 256 / convType;
            paraConvRelu(rs, convNet, 16 * 256, 256, fire8_expand1x1_w, fire8_expand1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 256, 64, 1, 0, 16, 0, fire8_squeeze1x1_Allocation, fire8_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire8_expand1x1 (ms): ", logEnable);
            //Thread.sleep(sleepTime);
            paraConvRelu(rs, convNet, 3 * 3 * 16 * 256, 256, fire8_expand3x3_w, fire8_expand3x3_b, 27 * 27 * parallelOFMs, 3,
                    27, 27, 27, 27, 256, 64, 1, 1, 16, 27 * 27 * 256, fire8_squeeze1x1_Allocation, fire8_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire8_expand3x3 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[27 * 27 * 512];
                fire8_expand_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire8_concat.bin", tmp, 27 * 27 * 512);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire8_expand finished.");

        Allocation pool8Allocation = Allocation.createSized(rs, Element.F32(rs), 13 * 13 * 512);
        paraMaxPool(rs, convNet, 13 * 13 * 128, 3, 27, 27, 2, fire8_expand_Allocation, pool8Allocation, "SqueezeNet/Logs", "4.txt", "Pool8 (ms): ", logEnable);
        if (debug) {
            float[] tmp = new float[13 * 13 * 512];
            pool8Allocation.copyTo(tmp);
            binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/4", "pool8.bin", tmp, 13 * 13 * 512);
        }
        Log.i(tag, "Pool8 finished.");

        Allocation fire9_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 13 * 13 * 64);
        int[] fire9_squeeze1x1_TypeSet = {1, 2, 4, 8, 16};
        for (int convType : fire9_squeeze1x1_TypeSet) {
            int parallelOFMs = 64 / convType;
            paraConvRelu(rs, convNet, 128 * 64, 64, fire9_squeeze1x1_w, fire9_squeeze1x1_b, 13 * 13 * parallelOFMs, 1,
                    13, 13, 13, 13, 64, 512, 1, 0, 128, 0, pool8Allocation, fire9_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire9_squeeze1x1 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[13 * 13 * 64];
                fire9_squeeze1x1_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire9_squeeze1x1.bin", tmp, 13 * 13 * 64);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire9_squeeze1x1 finished.");

        Allocation fire9_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 13 * 13 * 512);
        int[] fire9_expand_TypeSet = {1, 2, 4, 8, 16, 32, 64};
        for (int convType : fire9_expand_TypeSet) {
            int parallelOFMs = 256 / convType;
            paraConvRelu(rs, convNet, 16 * 256, 256, fire9_expand1x1_w, fire9_expand1x1_b, 13 * 13 * parallelOFMs, 1,
                    13, 13, 13, 13, 256, 64, 1, 0, 16, 0, fire9_squeeze1x1_Allocation, fire9_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire9_expand1x1 (ms): ", logEnable);
            //Thread.sleep(sleepTime);
            paraConvRelu(rs, convNet, 3 * 3 * 16 * 256, 256, fire9_expand3x3_w, fire9_expand3x3_b, 13 * 13 * parallelOFMs, 3,
                    13, 13, 13, 13, 256, 64, 1, 1, 16, 13 * 13 * 256, fire9_squeeze1x1_Allocation, fire9_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire9_expand3x3 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[13 * 13 * 512];
                fire9_expand_Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "fire9_concat.bin", tmp, 13 * 13 * 512);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Fire9_expand finished.");

        Allocation conv10Allocation = Allocation.createSized(rs, Element.F32(rs), 15 * 15 * 1000);
        int[] conv10TypeSet = {1};
        for (int convType : conv10TypeSet) {
            int parallelOFMs = 1000 / convType;
            paraConvRelu(rs, convNet, 128 * 1000, 1000, conv10_w, conv10_b, 15 * 15 * parallelOFMs, 1,
                    15, 15, 13, 13, 1000, 512, 1, 1, 128, 0, fire9_expand_Allocation, conv10Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "Conv10 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[15 * 15 * 1000];
                conv10Allocation.copyTo(tmp);
                binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "conv10.bin", tmp, 15 * 15 * 1000);
            }
            //Thread.sleep(sleepTime);
        }
        Log.i(tag, "Conv10 finished.");

        Allocation pool10Allocation = Allocation.createSized(rs, Element.F32(rs), 1000);
        paraAvgPool(rs, convNet, 250, 15, 15, 15, 1, conv10Allocation, pool10Allocation, "SqueezeNet/Logs", "4.txt", "Pool10 (ms): ", logEnable);
        if (debug) {
            float[] tmp = new float[1000];
            pool10Allocation.copyTo(tmp);
            binaryDumper("SqueezeNet/Intermediate_rslts/Parallel/4", "pool10.bin", tmp, 1000);
        }
        Log.i(tag, "Pool10 finished.");

        float[] prob = new float[1000];
        pool10Allocation.copyTo(prob);
        softmax(prob, prob, 1000, "SqueezeNet/Logs", "parallel.txt", "Prob from parallel (ms)", logEnable);
        if (result)
            binaryDumper("SqueezeNet/Intermediate_rslts/Parallel", "prob.bin", prob, 1000);
        Log.i(tag, "Prob finished.");
    }
    public float SqueezeNetInference (Context context) {
        loadParameters("SqueezeNet/Logs", "5000.txt", "/SqueezeNet/Parameters/Vectorized", logEnable);

        RenderScript rs = RenderScript.create(context);
        ScriptC_convNet convNet = new ScriptC_convNet(rs);

        int[] lbls = new int[5000];
        IntegerLoader(5000, "/SqueezeNet/Images/labels.bin", lbls);

        int correct = 0;

        for (int imgCount = 0; imgCount < 5000; imgCount++) {
            loader(227 * 227 * 3, "/SqueezeNet/Images/" + String.valueOf(imgCount + 1) + ".bin", img);

            Type.Builder imgType = new Type.Builder(rs, Element.F32_4(rs)).setX(227 * 227);
            Allocation imgAllocation = Allocation.createTyped(rs, imgType.create());
            paraReshape(rs, convNet, imgAllocation);

            int convType = 4;

            Allocation conv1Allocation = Allocation.createSized(rs, Element.F32(rs), 111 * 111 * 96);
            int parallelOFMs = 96 / convType;
            paraConvRelu(rs, convNet, 7 * 7 * 96, 96, conv1_w, conv1_b, 111 * 111 * parallelOFMs, 7,
                    111, 111, 227, 227, 96, 3, 2, 0, 1, 0, imgAllocation, conv1Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "Conv1 (ms): ", logEnable);


            Allocation pool1Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 96);
            paraMaxPool(rs, convNet, 55 * 55 * 24, 3, 111, 111, 2, conv1Allocation, pool1Allocation, "SqueezeNet/Logs", "4.txt", "Pool1 (ms): ", logEnable);

            Allocation fire2_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 16);
            parallelOFMs = 16 / convType;
            paraConvRelu(rs, convNet, 24 * 16, 16, fire2_squeeze1x1_w, fire2_squeeze1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 16, 96, 1, 0, 24, 0, pool1Allocation, fire2_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire2_squeeze1x1 (ms): ", logEnable);

            Allocation fire2_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 128);
            parallelOFMs = 64 / convType;
            paraConvRelu(rs, convNet, 4 * 64, 64, fire2_expand1x1_w, fire2_expand1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 64, 16, 1, 0, 4, 0, fire2_squeeze1x1_Allocation, fire2_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire2_expand1x1 (ms): ", logEnable);

            paraConvRelu(rs, convNet, 3 * 3 * 4 * 64, 64, fire2_expand3x3_w, fire2_expand3x3_b, 55 * 55 * parallelOFMs, 3,
                    55, 55, 55, 55, 64, 16, 1, 1, 4, 55 * 55 * 64, fire2_squeeze1x1_Allocation, fire2_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire2_expand3x3 (ms): ", logEnable);

            Allocation fire3_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 16);
            parallelOFMs = 16 / convType;
            paraConvRelu(rs, convNet, 32 * 16, 16, fire3_squeeze1x1_w, fire3_squeeze1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 16, 128, 1, 0, 32, 0, fire2_expand_Allocation, fire3_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire3_squeeze1x1 (ms): ", logEnable);

            Allocation fire3_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 128);
            parallelOFMs = 64 / convType;
            paraConvRelu(rs, convNet, 4 * 64, 64, fire3_expand1x1_w, fire3_expand1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 64, 16, 1, 0, 4, 0, fire3_squeeze1x1_Allocation, fire3_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire3_expand1x1 (ms): ", logEnable);

            paraConvRelu(rs, convNet, 3 * 3 * 4 * 64, 64, fire3_expand3x3_w, fire3_expand3x3_b, 55 * 55 * parallelOFMs, 3,
                    55, 55, 55, 55, 64, 16, 1, 1, 4, 55 * 55 * 64, fire3_squeeze1x1_Allocation, fire3_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire3_expand3x3 (ms): ", logEnable);

            Allocation fire4_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 32);
            parallelOFMs = 32 / convType;
            paraConvRelu(rs, convNet, 32 * 32, 32, fire4_squeeze1x1_w, fire4_squeeze1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 32, 128, 1, 0, 32, 0, fire3_expand_Allocation, fire4_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire4_squeeze1x1 (ms): ", logEnable);

            Allocation fire4_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 256);
            parallelOFMs = 128 / convType;
            paraConvRelu(rs, convNet, 8 * 128, 128, fire4_expand1x1_w, fire4_expand1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 128, 32, 1, 0, 8, 0, fire4_squeeze1x1_Allocation, fire4_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire4_expand1x1 (ms): ", logEnable);

            paraConvRelu(rs, convNet, 3 * 3 * 8 * 128, 128, fire4_expand3x3_w, fire4_expand3x3_b, 55 * 55 * parallelOFMs, 3,
                    55, 55, 55, 55, 128, 32, 1, 1, 8, 55 * 55 * 128, fire4_squeeze1x1_Allocation, fire4_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire4_expand3x3 (ms): ", logEnable);

            Allocation pool4Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 256);
            paraMaxPool(rs, convNet, 27 * 27 * 64, 3, 55, 55, 2, fire4_expand_Allocation, pool4Allocation, "SqueezeNet/Logs", "4.txt", "Pool4 (ms): ", logEnable);

            Allocation fire5_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 32);
            parallelOFMs = 32 / convType;
            paraConvRelu(rs, convNet, 64 * 32, 32, fire5_squeeze1x1_w, fire5_squeeze1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 32, 256, 1, 0, 64, 0, pool4Allocation, fire5_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire5_squeeze1x1 (ms): ", logEnable);

            Allocation fire5_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 256);
            parallelOFMs = 128 / convType;
            paraConvRelu(rs, convNet, 8 * 128, 128, fire5_expand1x1_w, fire5_expand1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 128, 32, 1, 0, 8, 0, fire5_squeeze1x1_Allocation, fire5_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire5_expand1x1 (ms): ", logEnable);

            paraConvRelu(rs, convNet, 3 * 3 * 8 * 128, 128, fire5_expand3x3_w, fire5_expand3x3_b, 27 * 27 * parallelOFMs, 3,
                    27, 27, 27, 27, 128, 32, 1, 1, 8, 27 * 27 * 128, fire5_squeeze1x1_Allocation, fire5_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire5_expand3x3 (ms): ", logEnable);

            Allocation fire6_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 48);
            parallelOFMs = 48 / convType;
            paraConvRelu(rs, convNet, 64 * 48, 48, fire6_squeeze1x1_w, fire6_squeeze1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 48, 256, 1, 0, 64, 0, fire5_expand_Allocation, fire6_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire6_squeeze1x1 (ms): ", logEnable);

            Allocation fire6_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 384);
            parallelOFMs = 192 / convType;
            paraConvRelu(rs, convNet, 12 * 192, 192, fire6_expand1x1_w, fire6_expand1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 192, 48, 1, 0, 12, 0, fire6_squeeze1x1_Allocation, fire6_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire6_expand1x1 (ms): ", logEnable);

            paraConvRelu(rs, convNet, 3 * 3 * 12 * 192, 192, fire6_expand3x3_w, fire6_expand3x3_b, 27 * 27 * parallelOFMs, 3,
                    27, 27, 27, 27, 192, 48, 1, 1, 12, 27 * 27 * 192, fire6_squeeze1x1_Allocation, fire6_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire6_expand3x3 (ms): ", logEnable);

            Allocation fire7_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 48);
            parallelOFMs = 48 / convType;
            paraConvRelu(rs, convNet, 96 * 48, 48, fire7_squeeze1x1_w, fire7_squeeze1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 48, 384, 1, 0, 96, 0, fire6_expand_Allocation, fire7_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire7_squeeze1x1 (ms): ", logEnable);

            Allocation fire7_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 384);
            parallelOFMs = 192 / convType;
            paraConvRelu(rs, convNet, 12 * 192, 192, fire7_expand1x1_w, fire7_expand1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 192, 48, 1, 0, 12, 0, fire7_squeeze1x1_Allocation, fire7_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire7_expand1x1 (ms): ", logEnable);

            paraConvRelu(rs, convNet, 3 * 3 * 12 * 192, 192, fire7_expand3x3_w, fire7_expand3x3_b, 27 * 27 * parallelOFMs, 3,
                    27, 27, 27, 27, 192, 48, 1, 1, 12, 27 * 27 * 192, fire7_squeeze1x1_Allocation, fire7_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire7_expand3x3 (ms): ", logEnable);

            Allocation fire8_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 64);
            parallelOFMs = 64 / convType;
            paraConvRelu(rs, convNet, 96 * 64, 64, fire8_squeeze1x1_w, fire8_squeeze1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 64, 384, 1, 0, 96, 0, fire7_expand_Allocation, fire8_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire8_squeeze1x1 (ms): ", logEnable);

            Allocation fire8_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 512);
            parallelOFMs = 256 / convType;
            paraConvRelu(rs, convNet, 16 * 256, 256, fire8_expand1x1_w, fire8_expand1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 256, 64, 1, 0, 16, 0, fire8_squeeze1x1_Allocation, fire8_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire8_expand1x1 (ms): ", logEnable);

            paraConvRelu(rs, convNet, 3 * 3 * 16 * 256, 256, fire8_expand3x3_w, fire8_expand3x3_b, 27 * 27 * parallelOFMs, 3,
                    27, 27, 27, 27, 256, 64, 1, 1, 16, 27 * 27 * 256, fire8_squeeze1x1_Allocation, fire8_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire8_expand3x3 (ms): ", logEnable);

            Allocation pool8Allocation = Allocation.createSized(rs, Element.F32(rs), 13 * 13 * 512);
            paraMaxPool(rs, convNet, 13 * 13 * 128, 3, 27, 27, 2, fire8_expand_Allocation, pool8Allocation, "SqueezeNet/Logs", "4.txt", "Pool8 (ms): ", logEnable);

            Allocation fire9_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 13 * 13 * 64);
            parallelOFMs = 64 / convType;
            paraConvRelu(rs, convNet, 128 * 64, 64, fire9_squeeze1x1_w, fire9_squeeze1x1_b, 13 * 13 * parallelOFMs, 1,
                    13, 13, 13, 13, 64, 512, 1, 0, 128, 0, pool8Allocation, fire9_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire9_squeeze1x1 (ms): ", logEnable);

            Allocation fire9_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 13 * 13 * 512);
            parallelOFMs = 256 / convType;
            paraConvRelu(rs, convNet, 16 * 256, 256, fire9_expand1x1_w, fire9_expand1x1_b, 13 * 13 * parallelOFMs, 1,
                    13, 13, 13, 13, 256, 64, 1, 0, 16, 0, fire9_squeeze1x1_Allocation, fire9_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire9_expand1x1 (ms): ", logEnable);

            paraConvRelu(rs, convNet, 3 * 3 * 16 * 256, 256, fire9_expand3x3_w, fire9_expand3x3_b, 13 * 13 * parallelOFMs, 3,
                    13, 13, 13, 13, 256, 64, 1, 1, 16, 13 * 13 * 256, fire9_squeeze1x1_Allocation, fire9_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire9_expand3x3 (ms): ", logEnable);

            convType = 1;
            Allocation conv10Allocation = Allocation.createSized(rs, Element.F32(rs), 15 * 15 * 1000);
            parallelOFMs = 1000 / convType;
            paraConvRelu(rs, convNet, 128 * 1000, 1000, conv10_w, conv10_b, 15 * 15 * parallelOFMs, 1,
                    15, 15, 13, 13, 1000, 512, 1, 1, 128, 0, fire9_expand_Allocation, conv10Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "Conv10 (ms): ", logEnable);

            Allocation pool10Allocation = Allocation.createSized(rs, Element.F32(rs), 1000);
            paraAvgPool(rs, convNet, 250, 15, 15, 15, 1, conv10Allocation, pool10Allocation, "SqueezeNet/Logs", "4.txt", "Pool10 (ms): ", logEnable);

            float[] prob = new float[1000];
            pool10Allocation.copyTo(prob);
            softmax(prob, prob, 1000, "SqueezeNet/Logs", "parallel.txt", "Prob from parallel (ms)", logEnable);
            binaryDumper("SqueezeNet/Intermediate_rslts/Parallel", "prob111.bin", prob, 1000);
            float max = -1;
            int maxIndex = -1;
            for (int i = 0; i < 1000; i++) {
                if (max < prob[i]) {
                    max = prob[i];
                    maxIndex = i;
                }
            }
            if (lbls[imgCount] == maxIndex) {
                correct++;
            }
            Log.i(tag, String.valueOf(lbls[imgCount]) + "," + String.valueOf(maxIndex));
            Log.i(tag, "Sample:" + String.valueOf(imgCount) + ", Correct: " + String.valueOf(correct));
        }
        float accuracy = correct / 5000.0f;
        Log.i(tag, "Accuracy:" + String.valueOf(accuracy));
        return  accuracy;

    }
    private void paraAvgPool(RenderScript rs, ScriptC_convNet convNet, int exeSpaceSz, int K,
                             int Hin, int Win, int S, Allocation inputAllocation,
                             Allocation outputAllocation, String logFolder,
                             String logFile, String logMsg, boolean logEnable){
        long start = System.currentTimeMillis();

        convNet.set_K(K);
        convNet.set_Wout((Win - K) / S + 1);
        convNet.set_Hout((Hin - K) / S + 1);
        convNet.set_Hin(Hin);
        convNet.set_Win(Win);
        convNet.set_S(S);

        Allocation exeSpace = Allocation.createSized(rs, Element.F32(rs), exeSpaceSz);
        convNet.set_in(inputAllocation);
        convNet.set_output(outputAllocation);

        convNet.forEach_avgPool(exeSpace);

        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");
    }
    public void parSqueezeNetPower (Context context) throws InterruptedException {

        loadParameters("SqueezeNet/Logs", "parallel.txt", "/SqueezeNet/Parameters/Vectorized", logEnable);

        RenderScript rs = RenderScript.create(context);
        ScriptC_convNet convNet = new ScriptC_convNet(rs);

        Intent createDatabase = new Intent("com.quicinc.trepn.start_profiling");
        createDatabase.putExtra("com.quicinc.trepn.database_file", "Parallel");
        context.sendBroadcast(createDatabase);

        Intent startProfiling = new Intent("com.quicinc.trepn.start_profiling");
        context.sendBroadcast(startProfiling);
        Allocation conv10Allocation = Allocation.createSized(rs, Element.F32(rs), 15 * 15 * 1000);
        for (int imgCount = 0; imgCount < 100; imgCount++) {
            loader(227 * 227 * 3, "/SqueezeNet/Images/" + String.valueOf(imgCount + 1) + ".bin", img);

            Type.Builder imgType = new Type.Builder(rs, Element.F32_4(rs)).setX(227 * 227);
            Allocation imgAllocation = Allocation.createTyped(rs, imgType.create());
            paraReshape(rs, convNet, imgAllocation);

            int convType = 1;
            Allocation conv1Allocation = Allocation.createSized(rs, Element.F32(rs), 111 * 111 * 96);
            int parallelOFMs = 96 / convType;
            paraConvRelu(rs, convNet, 7 * 7 * 96, 96, conv1_w, conv1_b, 111 * 111 * parallelOFMs, 7,
                    111, 111, 227, 227, 96, 3, 2, 0, 1, 0, imgAllocation, conv1Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "Conv1 (ms): ", logEnable);


            Allocation pool1Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 96);
            paraMaxPool(rs, convNet, 55 * 55 * 24, 3, 111, 111, 2, conv1Allocation, pool1Allocation, "SqueezeNet/Logs", "4.txt", "Pool1 (ms): ", logEnable);

            convType = 1;
            Allocation fire2_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 16);
            parallelOFMs = 16 / convType;
            paraConvRelu(rs, convNet, 24 * 16, 16, fire2_squeeze1x1_w, fire2_squeeze1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 16, 96, 1, 0, 24, 0, pool1Allocation, fire2_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire2_squeeze1x1 (ms): ", logEnable);

            convType = 1;
            Allocation fire2_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 128);
            parallelOFMs = 64 / convType;
            paraConvRelu(rs, convNet, 4 * 64, 64, fire2_expand1x1_w, fire2_expand1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 64, 16, 1, 0, 4, 0, fire2_squeeze1x1_Allocation, fire2_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire2_expand1x1 (ms): ", logEnable);

            convType = 16;
            parallelOFMs = 64 / convType;
            paraConvRelu(rs, convNet, 3 * 3 * 4 * 64, 64, fire2_expand3x3_w, fire2_expand3x3_b, 55 * 55 * parallelOFMs, 3,
                    55, 55, 55, 55, 64, 16, 1, 1, 4, 55 * 55 * 64, fire2_squeeze1x1_Allocation, fire2_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire2_expand3x3 (ms): ", logEnable);

            convType = 1;
            parallelOFMs = 16 / convType;
            Allocation fire3_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 16);
            paraConvRelu(rs, convNet, 32 * 16, 16, fire3_squeeze1x1_w, fire3_squeeze1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 16, 128, 1, 0, 32, 0, fire2_expand_Allocation, fire3_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire3_squeeze1x1 (ms): ", logEnable);

            convType = 1;
            parallelOFMs = 64 / convType;
            Allocation fire3_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 128);

            paraConvRelu(rs, convNet, 4 * 64, 64, fire3_expand1x1_w, fire3_expand1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 64, 16, 1, 0, 4, 0, fire3_squeeze1x1_Allocation, fire3_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire3_expand1x1 (ms): ", logEnable);

            convType = 16;
            parallelOFMs = 64 / convType;
            paraConvRelu(rs, convNet, 3 * 3 * 4 * 64, 64, fire3_expand3x3_w, fire3_expand3x3_b, 55 * 55 * parallelOFMs, 3,
                    55, 55, 55, 55, 64, 16, 1, 1, 4, 55 * 55 * 64, fire3_squeeze1x1_Allocation, fire3_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire3_expand3x3 (ms): ", logEnable);

            convType = 8;
            parallelOFMs = 32 / convType;
            Allocation fire4_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 32);
            paraConvRelu(rs, convNet, 32 * 32, 32, fire4_squeeze1x1_w, fire4_squeeze1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 32, 128, 1, 0, 32, 0, fire3_expand_Allocation, fire4_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire4_squeeze1x1 (ms): ", logEnable);

            convType = 1;
            parallelOFMs = 128 / convType;
            Allocation fire4_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 55 * 55 * 256);
            paraConvRelu(rs, convNet, 8 * 128, 128, fire4_expand1x1_w, fire4_expand1x1_b, 55 * 55 * parallelOFMs, 1,
                    55, 55, 55, 55, 128, 32, 1, 0, 8, 0, fire4_squeeze1x1_Allocation, fire4_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire4_expand1x1 (ms): ", logEnable);

            convType = 16;
            parallelOFMs = 128 / convType;
            paraConvRelu(rs, convNet, 3 * 3 * 8 * 128, 128, fire4_expand3x3_w, fire4_expand3x3_b, 55 * 55 * parallelOFMs, 3,
                    55, 55, 55, 55, 128, 32, 1, 1, 8, 55 * 55 * 128, fire4_squeeze1x1_Allocation, fire4_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire4_expand3x3 (ms): ", logEnable);

            Allocation pool4Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 256);
            paraMaxPool(rs, convNet, 27 * 27 * 64, 3, 55, 55, 2, fire4_expand_Allocation, pool4Allocation, "SqueezeNet/Logs", "4.txt", "Pool4 (ms): ", logEnable);

            convType = 1;
            parallelOFMs = 32 / convType;
            Allocation fire5_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 32);
            paraConvRelu(rs, convNet, 64 * 32, 32, fire5_squeeze1x1_w, fire5_squeeze1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 32, 256, 1, 0, 64, 0, pool4Allocation, fire5_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire5_squeeze1x1 (ms): ", logEnable);

            convType = 1;
            parallelOFMs = 128 / convType;
            Allocation fire5_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 256);
            paraConvRelu(rs, convNet, 8 * 128, 128, fire5_expand1x1_w, fire5_expand1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 128, 32, 1, 0, 8, 0, fire5_squeeze1x1_Allocation, fire5_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire5_expand1x1 (ms): ", logEnable);

            convType = 8;
            parallelOFMs = 128 / convType;
            paraConvRelu(rs, convNet, 3 * 3 * 8 * 128, 128, fire5_expand3x3_w, fire5_expand3x3_b, 27 * 27 * parallelOFMs, 3,
                    27, 27, 27, 27, 128, 32, 1, 1, 8, 27 * 27 * 128, fire5_squeeze1x1_Allocation, fire5_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire5_expand3x3 (ms): ", logEnable);

            convType = 1;
            parallelOFMs = 48 / convType;
            Allocation fire6_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 48);
            paraConvRelu(rs, convNet, 64 * 48, 48, fire6_squeeze1x1_w, fire6_squeeze1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 48, 256, 1, 0, 64, 0, fire5_expand_Allocation, fire6_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire6_squeeze1x1 (ms): ", logEnable);

            convType = 1;
            parallelOFMs = 192 / convType;
            Allocation fire6_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 384);
            paraConvRelu(rs, convNet, 12 * 192, 192, fire6_expand1x1_w, fire6_expand1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 192, 48, 1, 0, 12, 0, fire6_squeeze1x1_Allocation, fire6_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire6_expand1x1 (ms): ", logEnable);

            convType = 4;
            parallelOFMs = 192 / convType;
            paraConvRelu(rs, convNet, 3 * 3 * 12 * 192, 192, fire6_expand3x3_w, fire6_expand3x3_b, 27 * 27 * parallelOFMs, 3,
                    27, 27, 27, 27, 192, 48, 1, 1, 12, 27 * 27 * 192, fire6_squeeze1x1_Allocation, fire6_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire6_expand3x3 (ms): ", logEnable);

            convType = 1;
            parallelOFMs = 48 / convType;
            Allocation fire7_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 48);
            paraConvRelu(rs, convNet, 96 * 48, 48, fire7_squeeze1x1_w, fire7_squeeze1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 48, 384, 1, 0, 96, 0, fire6_expand_Allocation, fire7_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire7_squeeze1x1 (ms): ", logEnable);

            convType = 1;
            parallelOFMs = 192 / convType;
            Allocation fire7_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 384);
            paraConvRelu(rs, convNet, 12 * 192, 192, fire7_expand1x1_w, fire7_expand1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 192, 48, 1, 0, 12, 0, fire7_squeeze1x1_Allocation, fire7_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire7_expand1x1 (ms): ", logEnable);
            convType = 6;
            parallelOFMs = 192 / convType;
            paraConvRelu(rs, convNet, 3 * 3 * 12 * 192, 192, fire7_expand3x3_w, fire7_expand3x3_b, 27 * 27 * parallelOFMs, 3,
                    27, 27, 27, 27, 192, 48, 1, 1, 12, 27 * 27 * 192, fire7_squeeze1x1_Allocation, fire7_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire7_expand3x3 (ms): ", logEnable);

            convType = 1;
            parallelOFMs = 64 / convType;
            Allocation fire8_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 64);
            paraConvRelu(rs, convNet, 96 * 64, 64, fire8_squeeze1x1_w, fire8_squeeze1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 64, 384, 1, 0, 96, 0, fire7_expand_Allocation, fire8_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire8_squeeze1x1 (ms): ", logEnable);
            convType = 1;
            parallelOFMs = 256 / convType;
            Allocation fire8_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 27 * 27 * 512);
            paraConvRelu(rs, convNet, 16 * 256, 256, fire8_expand1x1_w, fire8_expand1x1_b, 27 * 27 * parallelOFMs, 1,
                    27, 27, 27, 27, 256, 64, 1, 0, 16, 0, fire8_squeeze1x1_Allocation, fire8_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire8_expand1x1 (ms): ", logEnable);

            convType = 4;
            parallelOFMs = 256 / convType;
            paraConvRelu(rs, convNet, 3 * 3 * 16 * 256, 256, fire8_expand3x3_w, fire8_expand3x3_b, 27 * 27 * parallelOFMs, 3,
                    27, 27, 27, 27, 256, 64, 1, 1, 16, 27 * 27 * 256, fire8_squeeze1x1_Allocation, fire8_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire8_expand3x3 (ms): ", logEnable);

            Allocation pool8Allocation = Allocation.createSized(rs, Element.F32(rs), 13 * 13 * 512);
            paraMaxPool(rs, convNet, 13 * 13 * 128, 3, 27, 27, 2, fire8_expand_Allocation, pool8Allocation, "SqueezeNet/Logs", "4.txt", "Pool8 (ms): ", logEnable);

            convType = 1;
            parallelOFMs = 64 / convType;
            Allocation fire9_squeeze1x1_Allocation = Allocation.createSized(rs, Element.F32(rs), 13 * 13 * 64);
            paraConvRelu(rs, convNet, 128 * 64, 64, fire9_squeeze1x1_w, fire9_squeeze1x1_b, 13 * 13 * parallelOFMs, 1,
                    13, 13, 13, 13, 64, 512, 1, 0, 128, 0, pool8Allocation, fire9_squeeze1x1_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire9_squeeze1x1 (ms): ", logEnable);

            convType = 1;
            parallelOFMs = 256 / convType;
            Allocation fire9_expand_Allocation = Allocation.createSized(rs, Element.F32(rs), 13 * 13 * 512);
            paraConvRelu(rs, convNet, 16 * 256, 256, fire9_expand1x1_w, fire9_expand1x1_b, 13 * 13 * parallelOFMs, 1,
                    13, 13, 13, 13, 256, 64, 1, 0, 16, 0, fire9_squeeze1x1_Allocation, fire9_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire9_expand1x1 (ms): ", logEnable);

            convType = 8;
            parallelOFMs = 256 / convType;
            paraConvRelu(rs, convNet, 3 * 3 * 16 * 256, 256, fire9_expand3x3_w, fire9_expand3x3_b, 13 * 13 * parallelOFMs, 3,
                    13, 13, 13, 13, 256, 64, 1, 1, 16, 13 * 13 * 256, fire9_squeeze1x1_Allocation, fire9_expand_Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "fire9_expand3x3 (ms): ", logEnable);

            convType = 1;
            conv10Allocation = Allocation.createSized(rs, Element.F32(rs), 15 * 15 * 1000);
            parallelOFMs = 1000 / convType;
            paraConvRelu(rs, convNet, 128 * 1000, 1000, conv10_w, conv10_b, 15 * 15 * parallelOFMs, 1,
                    15, 15, 13, 13, 1000, 512, 1, 1, 128, 0, fire9_expand_Allocation, conv10Allocation, convType,
                    parallelOFMs, "SqueezeNet/Logs", String.valueOf(convType) + ".txt",
                    "Conv10 (ms): ", logEnable);
        }
        Intent stopProfiling = new Intent("com.quicinc.trepn.stop_profiling");
        context.sendBroadcast(stopProfiling);

        Allocation pool10Allocation = Allocation.createSized(rs, Element.F32(rs), 1000);
        paraAvgPool(rs, convNet, 250, 15, 15, 15, 1, conv10Allocation, pool10Allocation, "SqueezeNet/Logs", "4.txt", "Pool10 (ms): ", logEnable);

        float[] prob = new float[1000];
        pool10Allocation.copyTo(prob);
        softmax(prob, prob, 1000, "SqueezeNet/Logs", "parallel.txt", "Prob from parallel (ms)", logEnable);

        if (result)
            binaryDumper("SqueezeNet/Intermediate_rslts/Parallel", "prob.bin", prob, 1000);
        Log.i(tag, "Prob finished.");
    }
    private void paraMaxPool (RenderScript rs, ScriptC_convNet convNet, int exeSpaceSz, int K,
                              int Hin, int Win, int S, Allocation inputAllocation,
                              Allocation outputAllocation, String logFolder,
                              String logFile, String logMsg, boolean logEnable) {
        long start = System.currentTimeMillis();

        convNet.set_K(K);
        convNet.set_Wout((Win - K) / S + 1);
        convNet.set_Hout((Hin - K) / S + 1);
        convNet.set_Hin(Hin);
        convNet.set_Win(Win);
        convNet.set_S(S);

        Allocation exeSpace = Allocation.createSized(rs, Element.F32(rs), exeSpaceSz);
        convNet.set_in(inputAllocation);
        convNet.set_output(outputAllocation);

        convNet.forEach_maxPool(exeSpace);

        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");

    }
    private void paraConvRelu (RenderScript rs, ScriptC_convNet convNet, int wghtSzF4, int biasSz,
                           float[] wght, float[] bias, int exeSpaceSz, int K,
                           int Wout, int Hout, int Win, int Hin, int M, int N, int S, int pad,
                           int N_new, int offset, Allocation inputAllocation,
                           Allocation outputAllocation, int convType, int parallelOFMs,
                           String logFolder, String logFile,
                           String logMsg, boolean logEnable) {
        //Intent createDatabase = new Intent("com.quicinc.trepn.start_profiling");
        //createDatabase.putExtra("com.quicinc.trepn.database_file", dataBaseName);
        //context.sendBroadcast(createDatabase);

        //Intent startProfiling = new Intent("com.quicinc.trepn.start_profiling");
        //context.sendBroadcast(startProfiling);

        long start = System.currentTimeMillis();

        Type.Builder weightType = new Type.Builder(rs, Element.F32_4(rs)).setX(wghtSzF4);
        Allocation weightAllocation = Allocation.createTyped(rs, weightType.create());
        weightAllocation.copyFromUnchecked(wght);

        Allocation biasAllocation = Allocation.createSized(rs, Element.F32(rs), biasSz);
        biasAllocation.copy1DRangeFrom(0, biasSz, bias);

        Allocation exeSpace = Allocation.createSized(rs, Element.F32(rs), exeSpaceSz);

        convNet.set_K(K);
        convNet.set_Wout(Wout);
        convNet.set_Hout(Hout);
        convNet.set_Hin(Hin);
        convNet.set_Win(Win);
        convNet.set_M(M);
        convNet.set_N(N);
        convNet.set_S(S);
        convNet.set_pad(pad);
        convNet.set_N_new(N_new);
        convNet.set_offset(offset);
        convNet.set_parallelOFMs(parallelOFMs);

        convNet.set_weight(weightAllocation);
        convNet.set_bias(biasAllocation);
        convNet.set_in(inputAllocation);

        convNet.set_output(outputAllocation);

        switch (convType) {
            case 1: convNet.forEach_conv_1(exeSpace);
                break;
            case 2: convNet.forEach_conv_2(exeSpace);
                break;
            case 4: convNet.forEach_conv_4(exeSpace);
                break;
            case 6: convNet.forEach_conv_6(exeSpace);
                break;
            case 8: convNet.forEach_conv_8(exeSpace);
                break;
            case 12: convNet.forEach_conv_12(exeSpace);
                break;
            case 16: convNet.forEach_conv_16(exeSpace);
                break;
            case 20: convNet.forEach_conv_20(exeSpace);
                break;
            case 24: convNet.forEach_conv_24(exeSpace);
                break;
            case 28: convNet.forEach_conv_28(exeSpace);
                break;
            case 32: convNet.forEach_conv_32(exeSpace);
                break;
            case 36: convNet.forEach_conv_36(exeSpace);
                break;
            case 40: convNet.forEach_conv_40(exeSpace);
                break;
            case 44: convNet.forEach_conv_44(exeSpace);
                break;
            case 48: convNet.forEach_conv_48(exeSpace);
                break;
            case 64: convNet.forEach_conv_64(exeSpace);
                break;
            default:Log.e(tag, convType + " is an invalid size for convolution kernel.");
                break;
        }

        long end = System.currentTimeMillis();
        //Intent stopProfiling = new Intent("com.quicinc.trepn.stop_profiling");
        //context.sendBroadcast(stopProfiling);
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");
    }
    private void paraReshape(RenderScript rs, ScriptC_convNet convNet, Allocation outputAllocation) {
        Allocation imgAllocation = Allocation.createSized(rs, Element.F32(rs), 227 * 227 * 3);
        imgAllocation.copyFrom(img);

        convNet.set_Hout(227);
        convNet.set_Wout(227);

        Allocation exeSpace = Allocation.createSized(rs, Element.F32(rs), 227 * 227);

        convNet.set_in(imgAllocation);
        convNet.set_output(outputAllocation);
        convNet.forEach_reshape(exeSpace);
    }
    private void loadParameters(String logFolder, String logFile, String paramsDir, boolean logEnable) {
        long start = System.currentTimeMillis();

        int sz = 227 * 227 * 3;
        img = new float[sz];
        loader(sz, paramsDir + "/img.bin", img);


        sz = 7 * 7 * 4 * 96;
        conv1_w = new float[sz];
        loader(sz, paramsDir + "/conv1_w.bin", conv1_w);

        sz = 96;
        conv1_b = new float[sz];
        loader(sz, paramsDir + "/conv1_b.bin", conv1_b);

        sz = 96 * 16; // 1 * 1 * 96 * 16
        fire2_squeeze1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire2_squeeze1x1_w.bin", fire2_squeeze1x1_w);

        sz = 16;
        fire2_squeeze1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire2_squeeze1x1_b.bin", fire2_squeeze1x1_b);

        sz = 16 * 64; //1 * 1 * 16 * 64
        fire2_expand1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire2_expand1x1_w.bin", fire2_expand1x1_w);

        sz = 64;
        fire2_expand1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire2_expand1x1_b.bin", fire2_expand1x1_b);

        sz = 3 * 3 * 16 * 64;
        fire2_expand3x3_w = new float[sz];
        loader(sz, paramsDir + "/fire2_expand3x3_w.bin", fire2_expand3x3_w);

        sz = 64;
        fire2_expand3x3_b = new float[sz];
        loader(sz, paramsDir + "/fire2_expand3x3_b.bin", fire2_expand3x3_b);

        sz = 128 * 16; //1 * 1 * 128 * 16
        fire3_squeeze1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire3_squeeze1x1_w.bin", fire3_squeeze1x1_w);

        sz = 16;
        fire3_squeeze1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire3_squeeze1x1_b.bin", fire3_squeeze1x1_b);

        sz = 16 * 64;//1 * 1 * 16 * 64
        fire3_expand1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire3_expand1x1_w.bin", fire3_expand1x1_w);

        sz = 64;
        fire3_expand1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire3_expand1x1_b.bin", fire3_expand1x1_b);

        sz = 3 * 3 * 16 * 64;
        fire3_expand3x3_w = new float[sz];
        loader(sz, paramsDir + "/fire3_expand3x3_w.bin", fire3_expand3x3_w);

        sz = 64;
        fire3_expand3x3_b = new float[sz];
        loader(sz, paramsDir + "/fire3_expand3x3_b.bin", fire3_expand3x3_b);

        sz = 128 * 32;//1 * 1 * 128 * 32
        fire4_squeeze1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire4_squeeze1x1_w.bin", fire4_squeeze1x1_w);

        sz = 32;
        fire4_squeeze1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire4_squeeze1x1_b.bin", fire4_squeeze1x1_b);

        sz = 32 * 128;//1 * 1 * 32 * 128;
        fire4_expand1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire4_expand1x1_w.bin", fire4_expand1x1_w);

        sz = 128;
        fire4_expand1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire4_expand1x1_b.bin", fire4_expand1x1_b);

        sz = 3 * 3 * 32 * 128;
        fire4_expand3x3_w = new float[sz];
        loader(sz, paramsDir + "/fire4_expand3x3_w.bin", fire4_expand3x3_w);

        sz = 128;
        fire4_expand3x3_b = new float[sz];
        loader(sz, paramsDir + "/fire4_expand3x3_b.bin", fire4_expand3x3_b);

        sz = 256 * 32;//1 * 1 * 256 * 32
        fire5_squeeze1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire5_squeeze1x1_w.bin", fire5_squeeze1x1_w);

        sz = 32;
        fire5_squeeze1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire5_squeeze1x1_b.bin", fire5_squeeze1x1_b);

        sz = 32 * 128; //1 * 1 * 32 * 128
        fire5_expand1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire5_expand1x1_w.bin", fire5_expand1x1_w);

        sz = 128;
        fire5_expand1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire5_expand1x1_b.bin", fire5_expand1x1_b);

        sz = 3 * 3 * 32 * 128;
        fire5_expand3x3_w = new float[sz];
        loader(sz, paramsDir + "/fire5_expand3x3_w.bin", fire5_expand3x3_w);

        sz = 128;
        fire5_expand3x3_b = new float[sz];
        loader(sz, paramsDir + "/fire5_expand3x3_b.bin", fire5_expand3x3_b);

        sz = 256 * 48; //1 * 1 * 256 * 48
        fire6_squeeze1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire6_squeeze1x1_w.bin", fire6_squeeze1x1_w);

        sz = 48;
        fire6_squeeze1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire6_squeeze1x1_b.bin", fire6_squeeze1x1_b);

        sz = 48 * 192; //1 * 1 * 48 * 192
        fire6_expand1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire6_expand1x1_w.bin", fire6_expand1x1_w);

        sz = 192;
        fire6_expand1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire6_expand1x1_b.bin", fire6_expand1x1_b);

        sz = 3 * 3 * 48 * 192;
        fire6_expand3x3_w = new float[sz];
        loader(sz, paramsDir + "/fire6_expand3x3_w.bin", fire6_expand3x3_w);

        sz = 192;
        fire6_expand3x3_b = new float[sz];
        loader(sz, paramsDir + "/fire6_expand3x3_b.bin", fire6_expand3x3_b);

        sz = 384 * 48; //1 * 1 * 384 * 48
        fire7_squeeze1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire7_squeeze1x1_w.bin", fire7_squeeze1x1_w);

        sz = 48;
        fire7_squeeze1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire7_squeeze1x1_b.bin", fire7_squeeze1x1_b);

        sz = 48 * 192; //1 * 1 * 48 * 192
        fire7_expand1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire7_expand1x1_w.bin", fire7_expand1x1_w);

        sz = 192;
        fire7_expand1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire7_expand1x1_b.bin", fire7_expand1x1_b);

        sz = 3 * 3 * 48 * 192;
        fire7_expand3x3_w = new float[sz];
        loader(sz, paramsDir + "/fire7_expand3x3_w.bin", fire7_expand3x3_w);

        sz = 192;
        fire7_expand3x3_b = new float[sz];
        loader(sz, paramsDir + "/fire7_expand3x3_b.bin", fire7_expand3x3_b);

        sz = 384 * 64;//1 * 1 * 384 * 64
        fire8_squeeze1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire8_squeeze1x1_w.bin", fire8_squeeze1x1_w);

        sz = 64;
        fire8_squeeze1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire8_squeeze1x1_b.bin", fire8_squeeze1x1_b);

        sz = 64 * 256; //1 * 1 * 64 * 256
        fire8_expand1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire8_expand1x1_w.bin", fire8_expand1x1_w);

        sz = 256;
        fire8_expand1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire8_expand1x1_b.bin", fire8_expand1x1_b);

        sz = 3 * 3 * 64 * 256;
        fire8_expand3x3_w = new float[sz];
        loader(sz, paramsDir + "/fire8_expand3x3_w.bin", fire8_expand3x3_w);

        sz = 256;
        fire8_expand3x3_b = new float[sz];
        loader(sz, paramsDir + "/fire8_expand3x3_b.bin", fire8_expand3x3_b);

        sz = 512 * 64;//1 * 1 * 512 * 64
        fire9_squeeze1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire9_squeeze1x1_w.bin", fire9_squeeze1x1_w);

        sz = 64;
        fire9_squeeze1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire9_squeeze1x1_b.bin", fire9_squeeze1x1_b);

        sz = 64 * 256;//1 * 1 * 64 * 256
        fire9_expand1x1_w = new float[sz];
        loader(sz, paramsDir + "/fire9_expand1x1_w.bin", fire9_expand1x1_w);

        sz = 256;
        fire9_expand1x1_b = new float[sz];
        loader(sz, paramsDir + "/fire9_expand1x1_b.bin", fire9_expand1x1_b);

        sz = 3 * 3 * 64 * 256;
        fire9_expand3x3_w = new float[sz];
        loader(sz, paramsDir + "/fire9_expand3x3_w.bin", fire9_expand3x3_w);

        sz = 256;
        fire9_expand3x3_b = new float[sz];
        loader(sz, paramsDir + "/fire9_expand3x3_b.bin", fire9_expand3x3_b);

        sz = 512 * 1000; //1 * 1 * 512 * 1000
        conv10_w = new float[sz];
        loader(sz, paramsDir + "/conv10_w.bin", conv10_w);

        sz = 1000;
        conv10_b = new float[sz];
        loader(sz, paramsDir + "/conv10_b.bin", conv10_b);

        long end = System.currentTimeMillis();


        if (logEnable)
            logWriter(logFolder, logFile, "Loading Parameters from SD Card (ms):\t" + String.valueOf(end - start) + "\n");
    }
    private void convRelu(float[] in, float[] out, float[] weight,
                      float[] bias, int Win, int Hin, int N,
                      int M, int K, int S, int pad, int group,
                      Context context, String logFolder,
                      String logFile, String dataBaseName,
                      String logMsg, boolean logEnable){

        long start = System.currentTimeMillis();

        int Wout, Hout, w, h, m, n, i, j;
        Wout = (Win + 2 * pad - K) / S + 1;
        Hout = (Hin + 2 * pad - K) / S + 1;
        //Convolve the input feature maps with the kernels.
        //The access to the input is shifted by the number of padding pixels.
        //Before every MAC, check if this is in the zero-padded area.
        //ToDo: Get rid of this initialization
        for (i = 0; i < Wout; i++){
            for (j = 0; j < Hout; j++) {
                for (int k = 0; k < M; k++) {
                    out[i * Hout * M + j * M + k] = 0;
                }
            }
        }
        switch (group){
            //The output depends on all input feature maps.
            case 1:
                for (w = 0; w<Wout; w++){
                    for (h = 0; h<Hout; h++){
                        for (m = 0; m<M; m++){
                            for (n = 0; n<N; n++){
                                for (i = 0; i<K; i++){
                                    for (j = 0; j<K; j++){
                                        if (w*S + i - pad<0 || w*S + i - pad >= Win ||
                                                h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
                                        out[(w)+(Wout*h) + (Wout*Hout*m)] +=
                                                in[(w*S + i - pad) + Win*(h*S + j - pad) + (Win*Hin*n)] *
                                                        weight[(i) + (K * j) + (K * K * n) + (K * K * N * m)];
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            //The input and output are split into 2 groups.
            case 2:
                //The first half of input feature maps depends on the first half of output feature maps.
                for (w = 0; w<Wout; w++){
                    for (h = 0; h<Hout; h++){
                        for (m = 0; m<M / 2; m++){
                            for (n = 0; n<N / 2; n++){
                                for (i = 0; i<K; i++){
                                    for (j = 0; j<K; j++){
                                        if (w*S + i - pad<0 || w*S + i - pad >= Win ||
                                                h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
                                        out[(w)+(Wout*h) + (Wout*Hout*m)] +=
                                                in[(w*S + i - pad) + Win*(h*S + j - pad) + (Win*Hin*n)] *
                                                        weight[(i)+(K*j) + (K*K*n) + (K*K*N / 2 * m)];
                                    }
                                }
                            }
                        }
                    }
                }
                //The second half of input feature maps depends on the second half of output feature maps.
                for (w = 0; w<Wout; w++){
                    for (h = 0; h<Hout; h++){
                        for (m = M / 2; m<M; m++){
                            for (n = N / 2; n<N; n++){
                                for (i = 0; i<K; i++){
                                    for (j = 0; j<K; j++){
                                        if (w*S + i - pad<0 || w*S + i - pad >= Win ||
                                                h*S + j - pad<0 || h*S + j - pad >= Hin) continue;
                                        out[(w)+(Wout*h) + (Wout*Hout*m)] +=
                                                in[(w*S + i - pad) + Win*(h*S + j - pad) + (Win*Hin*n)] *
                                                        weight[(i)+(K*j) + (K*K*(n - N / 2)) + (K*K*N / 2 * m)];
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            default:
                Log.e(tag, "ERROR: Convolution group must be 1 or 2.\n");
                break;
        }
        //Add the bias per output feature map.
        for (w = 0; w<Wout; w++){
            for (h = 0; h<Hout; h++){
                for (m = 0; m<M; m++){
                    out[(w)+(Wout*h) + (Wout*Hout*m)] += bias[m];
                }
            }
        }

        for (w = 0; w < Wout; w++){
            for (h = 0; h < Hout; h++){
                for (n = 0; n < M; n++){
                    out[w + Wout * h + Wout * Hout * n] = Math.max(0, out[w + Wout * h + Wout * Hout * n]);
                }
            }
        }
        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");
    }
    private void maxpool(float[] in, float[] out, int Win, int Hin,
                         int N, int kernelSize, int S,
                         Context context, String logFolder,
                         String logFile, String dataBaseName,
                         String logMsg, boolean logEnable){
         long start = System.currentTimeMillis();

        int w, ww, h, hh, n, Wout, Hout, Wstart, Wend, Hstart, Hend;
        float max;
        Wout = (Win - kernelSize) / S + 1;
        Hout = (Hin - kernelSize) / S + 1;
        for (w = 0; w<Wout; w++){
            for (h = 0; h<Hout; h++){
                for (n = 0; n<N; n++){
                    Wstart = w*S;
                    Hstart = h*S;
                    Wend = Wstart + kernelSize;
                    Hend = Hstart + kernelSize;
                    max = -Float.MAX_VALUE;
                    for (ww = Wstart; ww<Wend; ww++){
                        for (hh = Hstart; hh<Hend; hh++){
                            max = Math.max(max, in[ww + Win*hh + Win*Hin*n]);
                        }
                    }
                    out[w + Wout*h + Wout*Hout*n] = max;
                }
            }
        }

        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");
    }

    private void avgpool(float[] in, float[] out, int Win, int Hin,
                         int N, int kernelSize, int S,
                         Context context, String logFolder,
                         String logFile, String dataBaseName,
                         String logMsg, boolean logEnable) {
        long start = System.currentTimeMillis();

        int w, ww, h, hh, n, Wout, Hout, Wstart, Wend, Hstart, Hend;
        float sum = 0;
        Wout = (Win - kernelSize) / S + 1;
        Hout = (Hin - kernelSize) / S + 1;
        for (w = 0; w<Wout; w++){
            for (h = 0; h<Hout; h++){
                for (n = 0; n<N; n++){
                    Wstart = w*S;
                    Hstart = h*S;
                    Wend = Wstart + kernelSize;
                    Hend = Hstart + kernelSize;
                    sum = 0;
                    for (ww = Wstart; ww<Wend; ww++){
                        for (hh = Hstart; hh<Hend; hh++){
                            sum += in[ww + Win*hh + Win*Hin*n];
                        }
                    }
                    out[w + Wout*h + Wout*Hout*n] = sum / (kernelSize * kernelSize);
                }
            }
        }

        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");
    }
    private void softmax(float[] in, float[] out, int N,String logFolder, String logFile,
                         String logMsg, boolean logEnable){
        long start = System.currentTimeMillis();

        float maxIn, sum;
        int n;
        maxIn = -Float.MAX_VALUE;
        sum = 0;
        //Calculate max input.
        for (n = 0; n<N; n++){
            maxIn = Math.max(maxIn, in[n]);
        }
        //Calculate the exponential of each input.
        for (n = 0; n<N; n++){
            out[n] = (float) Math.exp(in[n] - maxIn);
        }
        //Calculate the sum of all exponentials.
        for (n = 0; n<N; n++){
            sum += out[n];
        }
        //Calculate the output.
        for (n = 0; n<N; n++){
            out[n] = out[n] / sum;
        }

        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");
    }
    private void loader(int size, String address, float [] array) {
        try {
            address = Environment.getExternalStorageDirectory().getPath() + address;
            RandomAccessFile file = new RandomAccessFile(address, "rw");
            FileChannel channel = file.getChannel();
            ByteBuffer buf = ByteBuffer.allocate(4 * size);
            buf.clear();
            channel.read(buf);
            buf.rewind();
            buf.order(ByteOrder.LITTLE_ENDIAN);
            buf.asFloatBuffer().get(array);
            channel.close();
            file.close();
        } catch (IOException e) {
            Log.e(tag, e.getMessage());
        }
    }
    private void IntegerLoader(int size, String address, int [] array) {
        try {
            address = Environment.getExternalStorageDirectory().getPath() + address;
            RandomAccessFile file = new RandomAccessFile(address, "rw");
            FileChannel channel = file.getChannel();
            ByteBuffer buf = ByteBuffer.allocate(4 * size);
            buf.clear();
            channel.read(buf);
            buf.rewind();
            buf.order(ByteOrder.LITTLE_ENDIAN);
            buf.asIntBuffer().get(array);
            channel.close();
            file.close();
        } catch (IOException e) {
            Log.e(tag, e.getMessage());
        }
    }
    private void logWriter (String dirs, String fileName, String msg) {
        File logDirectory = new File(Environment.getExternalStorageDirectory().getPath() + "/" + dirs);
        logDirectory.mkdirs();
        try {
            FileWriter log = new FileWriter(logDirectory + "/" + fileName, true);
            log.write(msg);
            log.close();
        } catch (IOException e) {
            Log.e(tag, "Error: Cannot write tag file: " + "/" + fileName + "\n" + e.getMessage());
        }
    }
    private void binaryDumper(String dirs, String fileName, float[] array, int size) {
        try {
            File dumpDirectory = new File(Environment.getExternalStorageDirectory().getPath() + "/" + dirs);
            dumpDirectory.mkdirs();

            RandomAccessFile file = new RandomAccessFile(dumpDirectory + "/" + fileName, "rw");
            FileChannel channel = file.getChannel();
            ByteBuffer buf = ByteBuffer.allocate(4 * size);
            buf.clear();

            //buf.order(ByteOrder.LITTLE_ENDIAN);
            buf.asFloatBuffer().put(array);
            channel.write(buf);
            buf.rewind();
            channel.close();
            file.close();
        } catch (IOException e) {
            Log.e(tag, e.getMessage());
        }
    }


}
