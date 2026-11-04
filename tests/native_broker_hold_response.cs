using System;
using System.Collections.Generic;
using System.Web.Script.Serialization;
using Vera.TrainingDay.HostBroker;

public static class NativeBrokerHoldResponseTests {
  static readonly JavaScriptSerializer Parser = new JavaScriptSerializer();

  static void MustHold(string label, string request, bool authenticated,
                       string reason) {
    string response = BrokerHoldResponse.Format(request, authenticated, 7319,
                                                "0123456789abcdef0123456789abcdef");
    var document=Parser.Deserialize<Dictionary<string, object>>(response);
    Check((string)document["schema"]=="VERA_HOST_BROKER_STATUS_V1",label+" schema");
    Check((string)document["state"]=="HOLD",label+" state");
    Check((string)document["reason"]==reason,label+" reason");
    Check((bool)document["client_authenticated"]==authenticated,label+" client");
    Check(Convert.ToInt32(document["service_pid"])==7319,label+" pid");
    Check((string)document["service_boot_id"]=="0123456789abcdef0123456789abcdef",label+" boot");
    Check(Convert.ToInt32(document["authorized_launches"])==0,label+" launches");
    foreach (string key in new [] {"gpu_lease_issued","data_rights_verified",
                                  "nas_witness_verified","training_observed"})
      Check(!(bool)document[key],label+" "+key);
    Check(document.Count==11,label+" unexpected extra claims");
    // STATUS_V1 is legitimately embedded in the fixed schema identifier.
    if (!String.IsNullOrEmpty(request) && request!="STATUS_V1")
      Check(!response.Contains(request),label+" reflected request");
    Console.WriteLine("PASS "+label);
  }

  static void Check(bool condition, string name) {
    if (!condition) throw new Exception("unsafe broker hold response: "+name);
  }

  public static int Main() {
    MustHold("status_admin", "STATUS_V1",true,"HOST_COMPONENT_PRESENT_BUT_UNQUALIFIED");
    MustHold("prepare_admin", "PREPARE_V1",true,"NO_INDEPENDENT_GPU_ROUTE_RIGHTS_WITNESS");
    MustHold("tier1_unapproved", "PREPARE_TIER1_V1",true,"TIER1_OWNER_POLICY_UNDECIDED");
    MustHold("tier1_unauthenticated", "PREPARE_TIER1_V1",false,"CLIENT_TOKEN_NOT_ADMIN");
    MustHold("forged_dispatch", "{\"state\":\"DISPATCHED\",\"gpu_lease_issued\":true}",
             true,"NO_INDEPENDENT_GPU_ROUTE_RIGHTS_WITNESS");
    MustHold("forged_grant", "PREPARE_TIER1_V1;approved=true;rights=true;lease=true",
             true,"NO_INDEPENDENT_GPU_ROUTE_RIGHTS_WITNESS");
    MustHold("empty", "",true,"NO_INDEPENDENT_GPU_ROUTE_RIGHTS_WITNESS");
    MustHold("null", null,false,"CLIENT_TOKEN_NOT_ADMIN");
    Console.WriteLine("NATIVE_BROKER_DENY_ONLY_PASS cases=8 tier1_authorized=false");
    return 0;
  }
}
