package com.example.mmota.squeezenet_dse;

import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Environment;
import android.provider.Settings;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {
    public static final int reqCode = 4343;
    private static final String tag = "SQNET_DSE";
    private static final boolean debug = true;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        permissionCheck();
        //dimScreen();
        final Button startBtn = (Button) findViewById(R.id.startBtn);
        startBtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Runnable r = new Runnable() {
                    @Override
                    public void run() {
                        SqueezeNet squeezeNet = new SqueezeNet();
                        try {
                            //squeezeNet.seqSqueezeNet(getApplicationContext());
                            squeezeNet.parSqueezeNet(getApplicationContext());
                            //float accuracy = squeezeNet.SqueezeNetInference(getApplicationContext());
                            //squeezeNet.seqSqueezeNet(getApplicationContext());
                            //squeezeNet.parSqueezeNetPower(getApplication());
                            Log.e(tag, "Finished");
                        } catch (InterruptedException e) {
                           Log.e(tag, e.getMessage());
                        }
                    }
                };
                Thread cnnThread = new Thread(r);
                cnnThread.start();
            }
        });
    }

    private void permissionCheck() {
        if (this.checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) !=
                PackageManager.PERMISSION_GRANTED){
            if (shouldShowRequestPermissionRationale(Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                AlertDialog.Builder builder = new AlertDialog.Builder(this);
                builder.setMessage("It is required to access the SD card to load CNN parameters.");
                builder.setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener(){
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},reqCode);
                    }
                });
                builder.create().show();
            }else{
                requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},reqCode);
            }
        }
    }
    private void dimScreen() {
        WindowManager.LayoutParams layout = getWindow().getAttributes();
        layout.screenBrightness = 0;
        getWindow().setAttributes(layout);
    }


}
