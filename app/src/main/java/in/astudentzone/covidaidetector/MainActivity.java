package in.astudentzone.covidaidetector;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.github.dhaval2404.imagepicker.ImagePicker;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Timer;
import java.util.TimerTask;

import in.astudentzone.covidaidetector.ml.Model;

public class MainActivity extends AppCompatActivity {

    private ImageView image;
    private Button predict;
    private LinearLayout layout;
    private ProgressBar normal,covid,viral;
    private TextView tv_normal,tv_covid,tv_viral,info_alert;
    private Bitmap img;

    public int result_normal,result_covid,result_pneu,counter1=0,counter2=0,counter3=0;

    private long backpressedtime;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        UIElements();


        image.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                layout.setVisibility(View.INVISIBLE);

                ImagePicker.with(MainActivity.this)
                        //.crop()	    			//Crop image(Optional), Check Customization for more option
                        //.compress(1024)			//Final image size will be less than 1 MB(Optional)
                        .maxResultSize(1080, 1080)	//Final image resolution will be less than 1080 x 1080(Optional)
                        .start(100);
            }
        });



        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                img=Bitmap.createScaledBitmap(img,224,224,true);
                try {
                    Model model = Model.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);

                    TensorImage tensorImage=new TensorImage(DataType.UINT8);
                    tensorImage.load(img);
                    ByteBuffer byteBuffer=tensorImage.getBuffer();

                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Model.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    // Releases model resources if no longer used.
                    model.close();

                    result_normal= (int) (outputFeature0.getFloatArray()[0]/255)*100;
                    result_covid= (int) (outputFeature0.getFloatArray()[1]/255)*100;
                    result_pneu=(int) (outputFeature0.getFloatArray()[2]/255)*100;


                    progress_bar();


                } catch (IOException e) {
                    // TODO Handle the exception
                }

            }
        });
        info_alert.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                alertDialogBox();
            }
        });
    }

    private void alertDialogBox() {
        String message="Upload high resolution image for best prediction.It is advisable not to rely completely on the result. \nIf you have doubts make sure to get a COVID test from a nearby hospital.\nStay safe and make other's safe.";
        AlertDialog alert=new AlertDialog.Builder(MainActivity.this)
                .setTitle("Message")
                .setMessage(message)
                .setCancelable(false)
                .setPositiveButton("OK,I Got It", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.dismiss();
                    }
                }).create();
        alert.show();
    }



    private void progress_bar() {
        layout.setVisibility(View.VISIBLE);

        normal.setProgress(result_normal);
        tv_normal.setText(String.valueOf(result_normal)+"%");

        covid.setProgress(result_covid);
        tv_covid.setText(String.valueOf(result_covid)+"%");

        viral.setProgress(result_pneu);
        tv_viral.setText(String.valueOf(result_pneu)+"%");
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable @org.jetbrains.annotations.Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode==100){
            image.setImageURI(data.getData());
            Uri uri=data.getData();
            try {
                img= MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
                predict.setVisibility(View.VISIBLE);
                info_alert.setVisibility(View.VISIBLE);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }else{
            predict.setVisibility(View.INVISIBLE);
            info_alert.setVisibility(View.INVISIBLE);
        }

    }

    private void UIElements() {
        image=findViewById(R.id.iv_image);
        predict=findViewById(R.id.btn_predict);
        info_alert=findViewById(R.id.info);
        layout=findViewById(R.id.bar_layout);
        normal=findViewById(R.id.bar_normal);
        covid=findViewById(R.id.bar_covid);
        viral=findViewById(R.id.bar_viral);
        tv_normal=findViewById(R.id.normal_per);
        tv_covid=findViewById(R.id.covid_per);
        tv_viral=findViewById(R.id.viral_per);
    }

    @Override
    public void onBackPressed() {

        if (backpressedtime+2000>System.currentTimeMillis()){
            super.onBackPressed();
            return;
        }else{
            Toast.makeText(getBaseContext(),"Press back again to exit",Toast.LENGTH_SHORT).show();
        }
        backpressedtime=System.currentTimeMillis();
    }
}