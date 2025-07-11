#include "C:/Users/rchen/Documents/Important Files/UIUC Research/Gazzola Lab/MEB/knot_pov_ray/default.inc"

camera{
    location <4,4,-4>
    angle 30
    look_at <0.1,0.4,-0.1>
    sky <0,1,0>
    right x*image_width/image_height
}
light_source{
    <15.0,10.5,-15.0>
    color rgb<0.09,0.09,0.1>
}
light_source{
    <1500,2500,-1000>
    color White
}

sphere_sweep {
    linear_spline 81
    ,<0.0,0.0,0.0>,0.025
    ,<-0.0010256157464158591,0.025784108724685482,-4.2004809391918166e-05>,0.025
    ,<-0.0017035279543670654,0.04634673471808801,-0.00012001251797763988>,0.025
    ,<-0.0027843888016718175,0.0735531105215765,-0.00032442133885575387>,0.025
    ,<-0.003540182367482671,0.09545037703757711,-0.0005910343682942863>,0.025
    ,<-0.0044710545182167745,0.12105566933636173,-0.0009366356593761576>,0.025
    ,<-0.005178507133060561,0.14382187589189474,-0.0013916087211281038>,0.025
    ,<-0.005920279303276457,0.16863542887596758,-0.0018557930620954244>,0.025
    ,<-0.006458121636102089,0.19180570345269518,-0.0024403400347099598>,0.025
    ,<-0.006937278733199492,0.2160887742996626,-0.0030135303987326137>,0.025
    ,<-0.007219115406689769,0.2393645649551802,-0.003680146672836086>,0.025
    ,<-0.007423423786373767,0.26330993231449257,-0.0043150389978921936>,0.025
    ,<-0.007449652970307615,0.28668767053296695,-0.004992631208004212>,0.025
    ,<-0.007369097151015617,0.31053026415112256,-0.005665818975170843>,0.025
    ,<-0.006992975233906696,0.3340182117397327,-0.006489414127690677>,0.025
    ,<-0.006273727588909081,0.3577970911914868,-0.007451712732891302>,0.025
    ,<-0.004747432988998053,0.38115312426955184,-0.00874030367035719>,0.025
    ,<-0.002063621276861394,0.40419989891854347,-0.010580959704082748>,0.025
    ,<0.0023864444067662753,0.426686939511261,-0.01258669814921744>,0.025
    ,<0.009356249724440588,0.4476462337518437,-0.015643931779811326>,0.025
    ,<0.020269275575732548,0.46377939467184626,-0.021763798898231917>,0.025
    ,<0.03511599319457829,0.4737225535743257,-0.028537468087603737>,0.025
    ,<0.05198458892263716,0.4787534130666639,-0.03126979801923381>,0.025
    ,<0.06838154586794863,0.48060787119758663,-0.028636154940708394>,0.025
    ,<0.08259906693159517,0.48051018132870826,-0.021267386068053885>,0.025
    ,<0.09324967850364059,0.47970348945742536,-0.010141266722443056>,0.025
    ,<0.1007356191825197,0.47900166076111234,0.0032579929048819523>,0.025
    ,<0.10595837943470386,0.4790444531275656,0.017651662480464625>,0.025
    ,<0.10948323890850695,0.4800839970621091,0.03244842569557337>,0.025
    ,<0.11111791158571657,0.4822333867867218,0.04735057454964824>,0.025
    ,<0.11097009756727874,0.4854872260433277,0.06209243113212586>,0.025
    ,<0.10888653253858789,0.48983946755003954,0.07631366373785627>,0.025
    ,<0.10498467099537996,0.4952176969535658,0.08970775343895025>,0.025
    ,<0.0991836267837967,0.5015191072865464,0.10184941215707727>,0.025
    ,<0.09166083731371417,0.5085943624056756,0.11238420704840434>,0.025
    ,<0.08247262897459212,0.5162249786919297,0.1208857370973232>,0.025
    ,<0.07192768657177502,0.5241652967758518,0.12706379230601736>,0.025
    ,<0.06026961612218239,0.5320987001855029,0.13065017613628344>,0.025
    ,<0.047932202695798565,0.539727388460233,0.1315805560686507>,0.025
    ,<0.035255987225381744,0.5467234223951982,0.1298753009476076>,0.025
    ,<0.022669062558612663,0.5528253588582278,0.12572519702908996>,0.025
    ,<0.010474310516847968,0.5577770724870821,0.11935668011807544>,0.025
    ,<-0.0009772413439852006,0.5613980606003284,0.1110658502311939>,0.025
    ,<-0.01144877220648029,0.5635329223615503,0.10113941951218554>,0.025
    ,<-0.0206757473608435,0.5641046585123929,0.0898988952588617>,0.025
    ,<-0.028523956077371185,0.5630835689381705,0.07765513868233938>,0.025
    ,<-0.03486271480707592,0.5605073963660392,0.06473559767491903>,0.025
    ,<-0.0396867421031776,0.5564510110587315,0.051434263727482096>,0.025
    ,<-0.0429766314220236,0.5510231960033065,0.03803792649371784>,0.025
    ,<-0.04479402667492137,0.5443514821916025,0.02477990065162549>,0.025
    ,<-0.04516630648073508,0.5365829834324138,0.011875560384501415>,0.025
    ,<-0.04417905322244849,0.5278795899741457,-0.0005215656236038289>,0.025
    ,<-0.0418845571084362,0.5184122476746849,-0.012275988205367435>,0.025
    ,<-0.03836537750774419,0.5083484650791976,-0.023294374441758577>,0.025
    ,<-0.033643362195785266,0.4978403048613077,-0.033468655487647075>,0.025
    ,<-0.02773112413418073,0.4870233913118001,-0.04267749996163989>,0.025
    ,<-0.0205562247013238,0.47599245849501204,-0.05075635871284603>,0.025
    ,<-0.012051906559638838,0.4648694035288723,-0.057435637948679355>,0.025
    ,<-0.002051795219417705,0.4538505228484577,-0.06210234516915958>,0.025
    ,<0.009643713319135598,0.44348492354510555,-0.06322358655958134>,0.025
    ,<0.022614392654353617,0.4349384647672899,-0.05904808216577235>,0.025
    ,<0.03541721335596578,0.42938344146919616,-0.049609202594224536>,0.025
    ,<0.046139085569302574,0.42798145761926765,-0.03580911937026137>,0.025
    ,<0.05325553666396917,0.43194995711171164,-0.019137493222092582>,0.025
    ,<0.056110309581897645,0.4420030306997789,-0.0027010001378069033>,0.025
    ,<0.05582637322743498,0.4582021097287963,0.010103721587294322>,0.025
    ,<0.05549149725342719,0.47885899116844677,0.017503746196934646>,0.025
    ,<0.056313978784930226,0.501025555240726,0.02094712528819918>,0.025
    ,<0.056976088496630065,0.5233285389205828,0.02296300361637105>,0.025
    ,<0.05721461030872849,0.5457988247182146,0.024216087023181853>,0.025
    ,<0.0573563384939984,0.5683686957732803,0.024864176903328497>,0.025
    ,<0.057362368061866245,0.5909910173403328,0.025228573676357603>,0.025
    ,<0.057335450211448945,0.6136711696440588,0.025397848058229866>,0.025
    ,<0.057271151175270055,0.6363405237667358,0.025493292686795614>,0.025
    ,<0.05715284807239193,0.6589659719829793,0.025569216438186777>,0.025
    ,<0.05697246382335707,0.6815338918601743,0.025623826052036162>,0.025
    ,<0.05672183639315331,0.7040399865897061,0.025647733584311293>,0.025
    ,<0.05641935864716018,0.726510022286633,0.0256142185312533>,0.025
    ,<0.056094396451604536,0.7489609422555167,0.025535375851796445>,0.025
    ,<0.055779039418918465,0.7713962354568996,0.025445906995275432>,0.025
    ,<0.05548696960436352,0.7938102263562531,0.025378379853040412>,0.025
    texture{
        pigment{ color rgb<0.45,0.39,1> transmit 0.000000 }
        finish{ phong 1 }
    }
    }
